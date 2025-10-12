# fragrance_ai/data/dataset_manager.py

import json
import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import sqlite3
import pandas as pd

from fragrance_ai.schemas.domain_models import (
    UserChoice, LearningHistory, OlfactoryDNA,
    ScentPhenotype, CreativeBrief
)


class DatasetManager:
    """Manager for training datasets and learning history"""

    def __init__(self, db_path: str = "data/fragrance_history.db"):
        """Initialize dataset manager with database"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for history logging"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # User interaction history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id_hash TEXT NOT NULL,  -- Hashed for privacy
                dna_id TEXT NOT NULL,
                phenotype_id TEXT NOT NULL,
                brief_id TEXT,

                -- Choice details
                chosen_option_id TEXT NOT NULL,
                presented_options TEXT NOT NULL,  -- JSON array
                rating REAL,
                feedback_text TEXT,

                -- Context
                iteration_number INTEGER DEFAULT 1,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                -- ML features (JSON)
                state_vector TEXT,
                action_vector TEXT,
                reward REAL,

                -- Indexing
                INDEX idx_user (user_id_hash),
                INDEX idx_session (session_id),
                INDEX idx_timestamp (timestamp)
            )
        """)

        # Experiment tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                user_id_hash TEXT NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,

                -- Configuration
                algorithm TEXT,  -- REINFORCE, PPO, GA
                hyperparameters TEXT,  -- JSON

                -- Metrics
                total_iterations INTEGER DEFAULT 0,
                avg_rating REAL,
                final_score REAL,

                -- Results
                final_recipe TEXT,  -- JSON
                status TEXT DEFAULT 'running',  -- running, completed, failed
                error_message TEXT
            )
        """)

        # Training logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                -- Metrics
                loss REAL,
                reward REAL,
                entropy REAL,
                policy_loss REAL,
                value_loss REAL,

                -- Violations
                ifra_violation_rate REAL,
                balance_score REAL,
                novelty_score REAL,

                -- Additional info (JSON)
                additional_metrics TEXT,

                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        conn.commit()
        conn.close()

    def hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy"""
        # Use SHA-256 with salt for better security
        salt = "fragrance_ai_2024"  # In production, use environment variable
        return hashlib.sha256(f"{salt}{user_id}".encode()).hexdigest()[:16]

    def log_interaction(self, choice: UserChoice, state_vector: Optional[List[float]] = None,
                       action_vector: Optional[List[float]] = None, reward: Optional[float] = None):
        """Log user interaction to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO user_interactions (
                session_id, user_id_hash, dna_id, phenotype_id, brief_id,
                chosen_option_id, presented_options, rating, feedback_text,
                iteration_number, timestamp, state_vector, action_vector, reward
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            choice.session_id,
            self.hash_user_id(choice.user_id),
            choice.dna_id,
            choice.phenotype_id,
            choice.brief_id,
            choice.chosen_option_id,
            json.dumps(choice.presented_options),
            choice.rating,
            choice.feedback_text,
            choice.iteration_number,
            choice.timestamp.isoformat(),
            json.dumps(state_vector) if state_vector else None,
            json.dumps(action_vector) if action_vector else None,
            reward
        ))

        conn.commit()
        conn.close()

    def start_experiment(self, user_id: str, algorithm: str = "PPO",
                        hyperparameters: Dict[str, Any] = None) -> str:
        """Start a new experiment and return experiment ID"""
        experiment_id = f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self.hash_user_id(user_id)[:8]}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO experiments (
                experiment_id, user_id_hash, algorithm, hyperparameters
            ) VALUES (?, ?, ?, ?)
        """, (
            experiment_id,
            self.hash_user_id(user_id),
            algorithm,
            json.dumps(hyperparameters) if hyperparameters else None
        ))

        conn.commit()
        conn.close()

        return experiment_id

    def log_training_step(self, experiment_id: str, step: int, metrics: Dict[str, float]):
        """Log training metrics for an experiment step"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Extract known metrics
        loss = metrics.get('loss')
        reward = metrics.get('reward')
        entropy = metrics.get('entropy')
        policy_loss = metrics.get('policy_loss')
        value_loss = metrics.get('value_loss')
        ifra_violation_rate = metrics.get('ifra_violation_rate')
        balance_score = metrics.get('balance_score')
        novelty_score = metrics.get('novelty_score')

        # Store remaining metrics as additional
        known_keys = {'loss', 'reward', 'entropy', 'policy_loss', 'value_loss',
                     'ifra_violation_rate', 'balance_score', 'novelty_score'}
        additional = {k: v for k, v in metrics.items() if k not in known_keys}

        cursor.execute("""
            INSERT INTO training_logs (
                experiment_id, step, loss, reward, entropy,
                policy_loss, value_loss, ifra_violation_rate,
                balance_score, novelty_score, additional_metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id, step, loss, reward, entropy,
            policy_loss, value_loss, ifra_violation_rate,
            balance_score, novelty_score,
            json.dumps(additional) if additional else None
        ))

        conn.commit()
        conn.close()

    def end_experiment(self, experiment_id: str, status: str = "completed",
                      final_recipe: Optional[Dict] = None, error_message: Optional[str] = None):
        """Mark experiment as ended"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Calculate final metrics
        cursor.execute("""
            SELECT AVG(rating) FROM user_interactions
            WHERE session_id IN (
                SELECT session_id FROM user_interactions
                WHERE user_id_hash = (
                    SELECT user_id_hash FROM experiments WHERE experiment_id = ?
                )
            )
        """, (experiment_id,))
        avg_rating = cursor.fetchone()[0]

        cursor.execute("""
            UPDATE experiments
            SET end_time = CURRENT_TIMESTAMP,
                status = ?,
                avg_rating = ?,
                final_recipe = ?,
                error_message = ?
            WHERE experiment_id = ?
        """, (
            status,
            avg_rating,
            json.dumps(final_recipe) if final_recipe else None,
            error_message,
            experiment_id
        ))

        conn.commit()
        conn.close()

    def get_user_history(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Get user's interaction history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        user_hash = self.hash_user_id(user_id)

        cursor.execute("""
            SELECT * FROM user_interactions
            WHERE user_id_hash = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_hash, limit))

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        conn.close()

        return [dict(zip(columns, row)) for row in rows]

    def get_experiment_metrics(self, experiment_id: str) -> pd.DataFrame:
        """Get training metrics for an experiment as DataFrame"""
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query("""
            SELECT * FROM training_logs
            WHERE experiment_id = ?
            ORDER BY step
        """, conn, params=(experiment_id,))

        conn.close()

        return df

    def export_dataset(self, output_path: str, start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None):
        """Export dataset for offline training"""
        conn = sqlite3.connect(self.db_path)

        # Build query with date filters
        query = "SELECT * FROM user_interactions"
        params = []

        if start_date or end_date:
            conditions = []
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date.isoformat())
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(end_date.isoformat())
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp"

        # Export to CSV
        df = pd.read_sql_query(query, conn, params=params)

        # Parse JSON columns
        json_columns = ['presented_options', 'state_vector', 'action_vector']
        for col in json_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if x else None)

        # Save to file
        output_path = Path(output_path)
        if output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        elif output_path.suffix == '.parquet':
            df.to_parquet(output_path, index=False)
        else:
            df.to_json(output_path, orient='records', lines=True)

        conn.close()

        return len(df)

    def calculate_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Calculate dataset statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Total interactions
        where_clause = ""
        params = ()
        if user_id:
            where_clause = "WHERE user_id_hash = ?"
            params = (self.hash_user_id(user_id),)

        cursor.execute(f"SELECT COUNT(*) FROM user_interactions {where_clause}", params)
        stats['total_interactions'] = cursor.fetchone()[0]

        # Average rating
        cursor.execute(f"SELECT AVG(rating) FROM user_interactions {where_clause} AND rating IS NOT NULL", params)
        stats['avg_rating'] = cursor.fetchone()[0]

        # Unique users
        cursor.execute("SELECT COUNT(DISTINCT user_id_hash) FROM user_interactions")
        stats['unique_users'] = cursor.fetchone()[0]

        # Experiments
        cursor.execute("SELECT COUNT(*) FROM experiments WHERE status = 'completed'")
        stats['completed_experiments'] = cursor.fetchone()[0]

        # Success rate
        cursor.execute("""
            SELECT AVG(CASE WHEN rating >= 4 THEN 1.0 ELSE 0.0 END)
            FROM user_interactions
            WHERE rating IS NOT NULL
        """)
        stats['success_rate'] = cursor.fetchone()[0]

        conn.close()

        return stats


# ============================================================================
# Data Loader for Training
# ============================================================================

class TrainingDataLoader:
    """Data loader for ML training"""

    def __init__(self, dataset_path: str):
        """Initialize with dataset path"""
        self.dataset_path = Path(dataset_path)
        self.data = None
        self._load_data()

    def _load_data(self):
        """Load dataset from file"""
        if self.dataset_path.suffix == '.csv':
            self.data = pd.read_csv(self.dataset_path)
        elif self.dataset_path.suffix == '.parquet':
            self.data = pd.read_parquet(self.dataset_path)
        else:
            self.data = pd.read_json(self.dataset_path, orient='records', lines=True)

    def get_training_batches(self, batch_size: int = 32, shuffle: bool = True):
        """Get batches for training"""
        if shuffle:
            data = self.data.sample(frac=1).reset_index(drop=True)
        else:
            data = self.data

        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i+batch_size]
            yield self._process_batch(batch)

    def _process_batch(self, batch: pd.DataFrame) -> Dict[str, Any]:
        """Process batch into training format"""
        # Parse JSON columns if they're strings
        for col in ['state_vector', 'action_vector', 'presented_options']:
            if col in batch.columns:
                batch[col] = batch[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )

        return {
            'states': batch['state_vector'].tolist(),
            'actions': batch['action_vector'].tolist(),
            'rewards': batch['reward'].tolist(),
            'ratings': batch['rating'].tolist(),
            'chosen_options': batch['chosen_option_id'].tolist()
        }

    def split_train_test(self, test_size: float = 0.2, temporal: bool = True):
        """Split data into train and test sets"""
        if temporal:
            # Temporal split (test on most recent data)
            split_idx = int(len(self.data) * (1 - test_size))
            train_data = self.data.iloc[:split_idx]
            test_data = self.data.iloc[split_idx:]
        else:
            # Random split
            from sklearn.model_selection import train_test_split
            train_data, test_data = train_test_split(
                self.data, test_size=test_size, random_state=42
            )

        return train_data, test_data


# ============================================================================
# Data Anonymization Utilities
# ============================================================================

class DataAnonymizer:
    """
    Advanced anonymization for RLHF datasets
    Removes all PII and personal identifiers
    """

    @staticmethod
    def anonymize_interaction_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove PII from interaction data

        Args:
            data: Raw interaction data

        Returns:
            Anonymized data with only safe fields
        """
        # Fields that are safe to keep
        safe_fields = {
            'interaction_id', 'session_id', 'user_id_hash', 'dna_id',
            'phenotype_id', 'brief_id', 'chosen_option_id',
            'presented_options', 'rating', 'iteration_number',
            'timestamp', 'state_vector', 'action_vector', 'reward'
        }

        # Remove PII fields
        pii_fields = {
            'user_id', 'email', 'phone', 'ip_address', 'user_agent',
            'device_id', 'location', 'real_name', 'address'
        }

        anonymized = {}

        for key, value in data.items():
            # Skip PII fields
            if key in pii_fields:
                continue

            # Keep safe fields
            if key in safe_fields:
                anonymized[key] = value

            # Hash any remaining ID-like fields
            elif key.endswith('_id') and key not in safe_fields:
                if isinstance(value, str):
                    anonymized[f"{key}_hash"] = hashlib.sha256(value.encode()).hexdigest()[:16]

        return anonymized

    @staticmethod
    def anonymize_batch(interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Anonymize batch of interactions"""
        return [DataAnonymizer.anonymize_interaction_data(inter) for inter in interactions]

    @staticmethod
    def anonymize_export(dataset_manager: 'DatasetManager', output_path: str):
        """
        Export fully anonymized dataset

        Args:
            dataset_manager: DatasetManager instance
            output_path: Output file path
        """
        conn = sqlite3.connect(dataset_manager.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all interactions
        cursor.execute("SELECT * FROM user_interactions ORDER BY timestamp")
        rows = cursor.fetchall()

        # Convert to dictionaries
        interactions = [dict(row) for row in rows]

        # Anonymize
        anonymized = DataAnonymizer.anonymize_batch(interactions)

        # Export
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dumps({
                "version": "1.0",
                "anonymized": True,
                "export_date": datetime.utcnow().isoformat(),
                "total_records": len(anonymized),
                "data": anonymized
            }, f, indent=2, ensure_ascii=False)

        conn.close()

        return len(anonymized)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'DatasetManager',
    'TrainingDataLoader',
    'DataAnonymizer'
]