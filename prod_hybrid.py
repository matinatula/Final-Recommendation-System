import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
from sqlalchemy import create_engine
from hybrid import HybridRecommender
from sqlalchemy.orm import sessionmaker

from database import SessionLocal, User, UserRecommendation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("recommendation_worker.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
MAX_WORKERS = 1  # Number of parallel tasks
RECOMMENDATIONS_PER_USER = 20  # Number of recommendations to generate per user
DEFAULT_EMOTION = "neutral"  # Default emotion filter
INTERVAL_HOURS = 6  # Run every 6 hours (adjust as needed)


class RecommendationWorker:
    """Worker class for automated recommendation generation"""

    def __init__(self, max_workers=MAX_WORKERS):
        self.max_workers = max_workers
        self.db_session = SessionLocal()
        self.hybrid_recommender = None

    def initialize_recommender(self):
        """Initialize and train the hybrid recommender system"""
        logger.info("Initializing hybrid recommender system...")
        try:
            self.hybrid_recommender = HybridRecommender(
                content_weight=0.3, collaborative_weight=0.5, emotion_weight=0.2
            )
            self.hybrid_recommender.load_and_train()
            logger.info("Hybrid recommender initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize recommender: {e}", exc_info=True)
            return False

    def get_all_users(self) -> List[str]:
        """Fetch all user IDs from database"""
        try:
            users = self.db_session.query(User.id).all()
            user_ids = [user.id for user in users]
            logger.info(f"Found {len(user_ids)} users in database")
            return user_ids
        except Exception as e:
            logger.error(f"Error fetching users: {e}", exc_info=True)
            return []

    def generate_recommendations_for_user(self, user_id: str) -> Dict:
        """Generate recommendations for a single user"""
        # Create a new session for this thread to avoid session conflicts
        session = SessionLocal()

        try:
            logger.info(f"Generating recommendations for user: {user_id}")

            # Generate recommendations using hybrid recommender
            recommendations = self.hybrid_recommender.recommend(
                user_id=user_id,
                target_emotion=DEFAULT_EMOTION,
                top_n=RECOMMENDATIONS_PER_USER,
            )

            if not recommendations:
                logger.warning(f"No recommendations generated for user: {user_id}")
                return {"user_id": user_id, "success": False, "count": 0}

            # Store recommendations in database
            stored_count = self.store_recommendations(session, user_id, recommendations)

            logger.info(
                f"Successfully generated {stored_count} recommendations for user: {user_id}"
            )
            return {"user_id": user_id, "success": True, "count": stored_count}

        except Exception as e:
            logger.error(
                f"Error generating recommendations for user {user_id}: {e}",
                exc_info=True,
            )
            return {"user_id": user_id, "success": False, "error": str(e)}
        finally:
            session.close()

    def store_recommendations(
        self, session, user_id: str, recommendations: List[Dict]
    ) -> int:
        """Store recommendations in the database using SQLAlchemy ORM"""
        try:
            # Clear old recommendations for this user
            deleted_count = (
                session.query(UserRecommendation)
                .filter(UserRecommendation.user_id == user_id)
                .delete()
            )

            logger.debug(
                f"Deleted {deleted_count} old recommendations for user {user_id}"
            )

            # Insert new recommendations
            stored_count = 0
            for rec in recommendations:
                # Generate reason based on scores
                reason_parts = []
                if rec["content_score"] > 0.5:
                    reason_parts.append("similar to your taste")
                if rec["collaborative_score"] > 0.5:
                    reason_parts.append("popular with similar users")
                if rec.get("emotion_match"):
                    reason_parts.append(f"matches {rec['emotion']} mood")

                reason = (
                    "Recommended: " + ", ".join(reason_parts)
                    if reason_parts
                    else "Based on your preferences"
                )

                # Create new recommendation record
                new_rec = UserRecommendation(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    track_id=rec["song_id"],
                    score=float(rec["final_score"]),
                    reason=reason,
                    created_at=datetime.now(timezone.utc),
                )
                session.add(new_rec)
                stored_count += 1

            # Commit all recommendations at once
            session.commit()
            logger.debug(
                f"Stored {stored_count} new recommendations for user {user_id}"
            )
            return stored_count

        except Exception as e:
            session.rollback()
            logger.error(
                f"Error storing recommendations for user {user_id}: {e}", exc_info=True
            )
            raise

    def process_users_parallel(self, user_ids: List[str]) -> Dict:
        """Process recommendations for multiple users in parallel"""
        logger.info(
            f"Starting parallel processing for {len(user_ids)} users with {self.max_workers} workers"
        )

        results = {
            "total_users": len(user_ids),
            "successful": 0,
            "failed": 0,
            "total_recommendations": 0,
            "start_time": datetime.now(timezone.utc),
            "failed_users": [],
        }

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_user = {
                executor.submit(
                    self.generate_recommendations_for_user, user_id
                ): user_id
                for user_id in user_ids
            }

            # Process completed tasks
            for future in as_completed(future_to_user):
                user_id = future_to_user[future]
                try:
                    result = future.result()
                    if result["success"]:
                        results["successful"] += 1
                        results["total_recommendations"] += result["count"]
                    else:
                        results["failed"] += 1
                        results["failed_users"].append(
                            {
                                "user_id": user_id,
                                "error": result.get("error", "Unknown error"),
                            }
                        )
                except Exception as e:
                    logger.error(
                        f"Exception processing user {user_id}: {e}", exc_info=True
                    )
                    results["failed"] += 1
                    results["failed_users"].append(
                        {"user_id": user_id, "error": str(e)}
                    )

        results["end_time"] = datetime.now(timezone.utc)
        results["duration_seconds"] = (
            results["end_time"] - results["start_time"]
        ).total_seconds()

        return results

    def run_recommendation_job(self):
        """Main job function - generates recommendations for all users"""
        logger.info("=" * 80)
        logger.info("RECOMMENDATION JOB STARTED")
        logger.info("=" * 80)

        start_time = datetime.now(timezone.utc)

        try:
            # Initialize recommender if not already done
            if not self.hybrid_recommender:
                if not self.initialize_recommender():
                    logger.error("Failed to initialize recommender. Aborting job.")
                    return

            # Get all users
            user_ids = self.get_all_users()
            if not user_ids:
                logger.warning("No users found. Aborting job.")
                return

            # Process users in parallel
            results = self.process_users_parallel(user_ids)

            # Log summary
            logger.info("=" * 80)
            logger.info("RECOMMENDATION JOB COMPLETED")
            logger.info(f"Total Users: {results['total_users']}")
            logger.info(f"Successful: {results['successful']}")
            logger.info(f"Failed: {results['failed']}")
            logger.info(
                f"Total Recommendations Generated: {results['total_recommendations']}"
            )
            logger.info(f"Duration: {results['duration_seconds']:.2f} seconds")

            # Log failed users if any
            if results["failed_users"]:
                logger.warning(f"Failed users ({len(results['failed_users'])}):")
                for failed in results["failed_users"][:10]:  # Show first 10
                    logger.warning(f"  - {failed['user_id']}: {failed['error']}")
                if len(results["failed_users"]) > 10:
                    logger.warning(
                        f"  ... and {len(results['failed_users']) - 10} more"
                    )

            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Fatal error in recommendation job: {e}", exc_info=True)

        finally:
            end_time = datetime.now(timezone.utc)
            logger.info(f"Job completed at {end_time}")

    def cleanup(self):
        """Clean up resources"""
        if self.hybrid_recommender:
            try:
                self.hybrid_recommender.close()
            except Exception as e:
                logger.error(f"Error closing hybrid recommender: {e}")

        if self.db_session:
            try:
                self.db_session.close()
            except Exception as e:
                logger.error(f"Error closing database session: {e}")

        logger.info("Worker cleanup completed")


def main():
    """Main function to run the scheduled worker"""
    logger.info("Starting Recommendation Worker Service")
    logger.info(
        f"Configuration: MAX_WORKERS={MAX_WORKERS}, RECOMMENDATIONS_PER_USER={RECOMMENDATIONS_PER_USER}, INTERVAL_HOURS={INTERVAL_HOURS}"
    )

    # Create worker instance
    worker = RecommendationWorker(max_workers=MAX_WORKERS)

    # Create scheduler
    scheduler = BlockingScheduler()

    # Schedule job to run continuously at regular intervals
    # Runs every 6 hours (adjust INTERVAL_HOURS constant to change frequency)
    scheduler.add_job(
        worker.run_recommendation_job,
        trigger="interval",
        hours=INTERVAL_HOURS,  # Run every N hours
        id="recommendation_job",
        name="Generate user recommendations",
        replace_existing=True,
        max_instances=1,  # Prevent overlapping jobs
    )

    logger.info("Scheduled jobs:")
    for job in scheduler.get_jobs():
        # Use trigger.get_next_fire_time(None, datetime.now(timezone.utc)) for next run time
        try:
            next_run = job.trigger.get_next_fire_time(None, datetime.now(timezone.utc))
        except Exception:
            next_run = None
        logger.info(
            f"  - {job.name} (ID: {job.id}) - Next run: {next_run if next_run else 'N/A'}"
        )

    # Run once immediately on startup (optional - comment out if not needed)
    logger.info("Running initial recommendation job on startup...")
    try:
        worker.run_recommendation_job()
    except Exception as e:
        logger.error(f"Initial job failed: {e}", exc_info=True)

    try:
        logger.info("Worker is running. Press Ctrl+C to exit.")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down scheduler...")
        scheduler.shutdown()
        worker.cleanup()
        logger.info("Worker service stopped")


if __name__ == "__main__":
    main()
