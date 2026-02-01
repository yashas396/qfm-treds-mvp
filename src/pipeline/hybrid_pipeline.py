"""
QGAI Quantum Financial Modeling - TReDS MVP
Hybrid Pipeline Module

This module provides the complete hybrid classical-quantum pipeline:
- Data loading and preprocessing
- Feature engineering
- Classical default prediction
- Quantum ring detection
- Composite risk scoring
- Result generation

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import get_config, Config
from src.data_generation import EntityGenerator, InvoiceGenerator
from src.feature_engineering import FeatureEngineer, FeatureEngineeringResult
from src.classical import DefaultPredictor, ModelTrainer, ModelTrainingResult
from src.quantum import RingDetector, RingDetectionResult
from .risk_scorer import RiskScorer, RiskScoringResult


@dataclass
class PipelineStage:
    """Information about a pipeline stage."""
    name: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: Dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class HybridPipelineResult:
    """Complete result from hybrid pipeline execution."""
    # Data
    entities_df: pd.DataFrame
    invoices_df: pd.DataFrame

    # Feature engineering
    feature_result: FeatureEngineeringResult

    # Classical results
    classical_result: ModelTrainingResult
    default_probabilities: np.ndarray

    # Quantum results
    quantum_result: RingDetectionResult
    ring_scores: Dict[str, float]

    # Combined results
    risk_result: RiskScoringResult

    # Metadata
    stages: List[PipelineStage]
    total_runtime_seconds: float
    pipeline_timestamp: datetime = field(default_factory=datetime.now)


class HybridPipeline:
    """
    Complete hybrid classical-quantum fraud detection pipeline.

    This class orchestrates all components:
    1. Data Generation (or loading)
    2. Feature Engineering
    3. Classical Default Prediction
    4. Quantum Ring Detection
    5. Composite Risk Scoring

    The pipeline can run end-to-end or stage-by-stage for debugging.

    Attributes:
        config: System configuration
        feature_engineer: FeatureEngineer instance
        predictor: DefaultPredictor instance
        detector: RingDetector instance
        scorer: RiskScorer instance

    Example:
        >>> pipeline = HybridPipeline()
        >>> result = pipeline.run()
        >>> print(f"High risk invoices: {result.risk_result.n_high_risk}")
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        k_communities: Optional[int] = None
    ):
        """
        Initialize HybridPipeline.

        Args:
            config: System configuration
            k_communities: Number of communities for ring detection.
                           If None, uses config.quantum.k_communities.
        """
        self.config = config or get_config()
        self.k_communities = k_communities if k_communities is not None else self.config.quantum.k_communities

        # Components will be initialized on run
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.predictor: Optional[DefaultPredictor] = None
        self.detector: Optional[RingDetector] = None
        self.scorer: Optional[RiskScorer] = None

        self._stages: List[PipelineStage] = []

    def run(
        self,
        entities_df: Optional[pd.DataFrame] = None,
        invoices_df: Optional[pd.DataFrame] = None,
        generate_data: bool = True,
        verbose: bool = True
    ) -> HybridPipelineResult:
        """
        Run complete hybrid pipeline.

        Args:
            entities_df: Optional pre-loaded entity data
            invoices_df: Optional pre-loaded invoice data
            generate_data: Whether to generate synthetic data if not provided
            verbose: Whether to print progress

        Returns:
            HybridPipelineResult: Complete pipeline result
        """
        start_time = datetime.now()
        self._stages = []

        # Stage 1: Data
        if entities_df is None or invoices_df is None:
            if generate_data:
                entities_df, invoices_df = self._run_data_generation(verbose)
            else:
                raise ValueError("Either provide data or set generate_data=True")
        else:
            self._add_stage('data_loading', 'completed',
                           metrics={'entities': len(entities_df), 'invoices': len(invoices_df)})

        # Stage 2: Feature Engineering
        feature_result = self._run_feature_engineering(entities_df, invoices_df, verbose)

        # Stage 3: Classical Model
        classical_result, default_probs = self._run_classical_model(feature_result, verbose)

        # Stage 4: Quantum Detection
        quantum_result, ring_scores = self._run_quantum_detection(feature_result, verbose)

        # Stage 5: Risk Scoring
        risk_result = self._run_risk_scoring(
            invoices_df, default_probs, ring_scores, verbose
        )

        # Calculate total runtime
        end_time = datetime.now()
        total_runtime = (end_time - start_time).total_seconds()

        if verbose:
            print(f"\nTotal pipeline runtime: {total_runtime:.2f} seconds")

        return HybridPipelineResult(
            entities_df=entities_df,
            invoices_df=invoices_df,
            feature_result=feature_result,
            classical_result=classical_result,
            default_probabilities=default_probs,
            quantum_result=quantum_result,
            ring_scores=ring_scores,
            risk_result=risk_result,
            stages=self._stages,
            total_runtime_seconds=total_runtime
        )

    def _run_data_generation(self, verbose: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run data generation stage."""
        stage = self._start_stage('data_generation')

        try:
            if verbose:
                print("\n[Stage 1] Generating synthetic data...")

            entity_gen = EntityGenerator()
            entities_df = entity_gen.generate()

            invoice_gen = InvoiceGenerator()
            invoices_df = invoice_gen.generate(entities_df)

            stage.metrics = {
                'n_entities': len(entities_df),
                'n_invoices': len(invoices_df),
                'n_ring_members': entities_df['is_ring_member'].sum()
            }

            if verbose:
                print(f"   Entities: {len(entities_df)}")
                print(f"   Invoices: {len(invoices_df)}")
                print(f"   Ring members: {stage.metrics['n_ring_members']}")

            self._complete_stage(stage)
            return entities_df, invoices_df

        except Exception as e:
            self._fail_stage(stage, str(e))
            raise

    def _run_feature_engineering(
        self,
        entities_df: pd.DataFrame,
        invoices_df: pd.DataFrame,
        verbose: bool
    ) -> FeatureEngineeringResult:
        """Run feature engineering stage."""
        stage = self._start_stage('feature_engineering')

        try:
            if verbose:
                print("\n[Stage 2] Engineering features...")

            self.feature_engineer = FeatureEngineer()
            result = self.feature_engineer.fit_transform(
                entities_df, invoices_df, build_graph=True
            )

            stage.metrics = {
                'n_features': result.n_features,
                'n_samples': result.n_samples,
                'n_graph_nodes': result.graph_result.n_nodes if result.graph_result else 0,
                'n_graph_edges': result.graph_result.n_edges if result.graph_result else 0
            }

            if verbose:
                print(f"   Features: {result.n_features}")
                print(f"   Samples: {result.n_samples}")
                if result.graph_result:
                    print(f"   Graph: {result.graph_result.n_nodes} nodes, {result.graph_result.n_edges} edges")

            self._complete_stage(stage)
            return result

        except Exception as e:
            self._fail_stage(stage, str(e))
            raise

    def _run_classical_model(
        self,
        feature_result: FeatureEngineeringResult,
        verbose: bool
    ) -> Tuple[ModelTrainingResult, np.ndarray]:
        """Run classical model training stage."""
        stage = self._start_stage('classical_model')

        try:
            if verbose:
                print("\n[Stage 3] Training classical model...")

            # Get feature matrix
            X, y = self.feature_engineer.get_feature_matrix(feature_result)

            # Train model
            trainer = ModelTrainer()
            result = trainer.train(X, y, feature_result.feature_names)

            self.predictor = result.model

            # Get predictions on all data
            default_probs = self.predictor.predict_proba(X)

            stage.metrics = {
                'auc_roc': result.test_evaluation.auc_roc,
                'precision': result.test_evaluation.precision,
                'recall': result.test_evaluation.recall,
                'f1': result.test_evaluation.f1,
                'cv_mean': result.training_result.evaluation.cv_mean
            }

            if verbose:
                print(f"   AUC-ROC: {stage.metrics['auc_roc']:.4f}")
                print(f"   CV AUC-ROC: {stage.metrics['cv_mean']:.4f}")
                print(f"   Mean P(default): {default_probs.mean():.4f}")

            self._complete_stage(stage)
            return result, default_probs

        except Exception as e:
            self._fail_stage(stage, str(e))
            raise

    def _run_quantum_detection(
        self,
        feature_result: FeatureEngineeringResult,
        verbose: bool
    ) -> Tuple[RingDetectionResult, Dict[str, float]]:
        """Run quantum ring detection stage."""
        stage = self._start_stage('quantum_detection')

        try:
            if verbose:
                print("\n[Stage 4] Running quantum ring detection...")

            self.detector = RingDetector(k_communities=self.k_communities)

            result = self.detector.detect(
                feature_result.graph_result.modularity_matrix,
                feature_result.graph_result.node_list,
                feature_result.graph_result.adjacency_matrix,
                num_reads=self.config.quantum.num_reads,
                num_sweeps=self.config.quantum.num_sweeps
            )

            ring_scores = self.detector.get_ring_scores(result)

            stage.metrics = {
                'n_communities': result.n_communities_found,
                'n_suspicious': result.n_suspicious_communities,
                'modularity': result.modularity,
                'constraint_violations': result.constraint_violations
            }

            if verbose:
                print(f"   Communities: {result.n_communities_found}")
                print(f"   Suspicious: {result.n_suspicious_communities}")
                print(f"   Modularity: {result.modularity:.4f}")

            self._complete_stage(stage)
            return result, ring_scores

        except Exception as e:
            self._fail_stage(stage, str(e))
            raise

    def _run_risk_scoring(
        self,
        invoices_df: pd.DataFrame,
        default_probs: np.ndarray,
        ring_scores: Dict[str, float],
        verbose: bool
    ) -> RiskScoringResult:
        """Run risk scoring stage."""
        stage = self._start_stage('risk_scoring')

        try:
            if verbose:
                print("\n[Stage 5] Computing risk scores...")

            self.scorer = RiskScorer()
            result = self.scorer.score(invoices_df, default_probs, ring_scores)

            summary = self.scorer.get_risk_summary(result)

            stage.metrics = {
                'n_high_risk': result.n_high_risk,
                'n_ring_associated': result.n_ring_associated,
                'pct_high_risk': summary['pct_high_risk'],
                'avg_composite_score': summary['avg_composite_score']
            }

            if verbose:
                print(f"   High risk invoices: {result.n_high_risk} ({summary['pct_high_risk']:.2f}%)")
                print(f"   Ring associated: {result.n_ring_associated}")
                print(f"   Avg composite score: {summary['avg_composite_score']:.4f}")

            self._complete_stage(stage)
            return result

        except Exception as e:
            self._fail_stage(stage, str(e))
            raise

    def _start_stage(self, name: str) -> PipelineStage:
        """Start a pipeline stage."""
        stage = PipelineStage(
            name=name,
            status='running',
            start_time=datetime.now()
        )
        self._stages.append(stage)
        return stage

    def _complete_stage(self, stage: PipelineStage) -> None:
        """Mark a stage as completed."""
        stage.status = 'completed'
        stage.end_time = datetime.now()

    def _fail_stage(self, stage: PipelineStage, error: str) -> None:
        """Mark a stage as failed."""
        stage.status = 'failed'
        stage.end_time = datetime.now()
        stage.error = error

    def _add_stage(
        self,
        name: str,
        status: str,
        metrics: Optional[Dict] = None
    ) -> None:
        """Add a stage record."""
        self._stages.append(PipelineStage(
            name=name,
            status=status,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=metrics or {}
        ))

    def get_stage_summary(self) -> pd.DataFrame:
        """Get summary of pipeline stages."""
        records = []
        for stage in self._stages:
            runtime = None
            if stage.start_time and stage.end_time:
                runtime = (stage.end_time - stage.start_time).total_seconds()

            records.append({
                'stage': stage.name,
                'status': stage.status,
                'runtime_seconds': runtime,
                **stage.metrics
            })

        return pd.DataFrame(records)


def run_hybrid_pipeline(
    entities_df: Optional[pd.DataFrame] = None,
    invoices_df: Optional[pd.DataFrame] = None,
    verbose: bool = True
) -> HybridPipelineResult:
    """
    Convenience function to run hybrid pipeline.

    Args:
        entities_df: Optional entity data
        invoices_df: Optional invoice data
        verbose: Whether to print progress

    Returns:
        HybridPipelineResult
    """
    pipeline = HybridPipeline()
    return pipeline.run(entities_df, invoices_df, generate_data=True, verbose=verbose)


if __name__ == "__main__":
    # Test hybrid pipeline
    print("=" * 70)
    print("HYBRID PIPELINE TEST")
    print("=" * 70)

    pipeline = HybridPipeline(k_communities=5)
    result = pipeline.run(verbose=True)

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    # Stage summary
    print("\nStage Summary:")
    print(pipeline.get_stage_summary().to_string())

    # Risk summary
    summary = pipeline.scorer.get_risk_summary(result.risk_result)
    print(f"\nRisk Summary:")
    print(f"  Total invoices: {summary['total_invoices']}")
    print(f"  High risk: {summary['n_high_risk']} ({summary['pct_high_risk']:.2f}%)")
    print(f"  Ring associated: {summary['n_ring_associated']} ({summary['pct_ring_associated']:.2f}%)")

    print(f"\n  Risk level distribution:")
    for level, count in summary['risk_level_counts'].items():
        print(f"    {level}: {count}")

    # Top high-risk invoices
    targeting = pipeline.scorer.get_targeting_list(result.risk_result, top_n=5)
    print(f"\n  Top 5 high-risk invoices:")
    print(targeting[['invoice_id', 'composite_score', 'risk_level']].to_string(index=False))

    # MVP Criteria Check
    print("\n" + "-" * 40)
    print("MVP CRITERIA CHECK:")

    auc_target = 0.75
    auc_actual = result.classical_result.test_evaluation.auc_roc
    print(f"  AUC-ROC >= {auc_target}: {auc_actual:.4f} [{'PASS' if auc_actual >= auc_target else 'FAIL'}]")

    print(f"  Modularity: {result.quantum_result.modularity:.4f}")
    print(f"  Pipeline completed: [PASS]")

    print("\n" + "=" * 70)
    print("HYBRID PIPELINE TEST COMPLETE")
    print("=" * 70)
