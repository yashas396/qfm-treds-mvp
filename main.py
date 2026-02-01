#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QGAI Quantum Financial Modeling - TReDS MVP
Main Entry Point

Hybrid Classical-Quantum TReDS Invoice Fraud Detection System

This script serves as the main entry point for running the complete
fraud detection pipeline including:
1. Synthetic data generation
2. Feature engineering
3. Classical default prediction
4. Quantum ring detection
5. Composite risk scoring
6. Report generation

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026

Usage:
    python main.py                          # Run with default configuration
    python main.py --mode train             # Train models only
    python main.py --mode predict           # Run predictions only
    python main.py --mode full              # Full pipeline (default)
    python main.py --config custom.yaml     # Use custom configuration
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    """Print application banner."""
    banner = """
+===============================================================================+
|                                                                               |
|    QGAI - Quantum Financial Modeling                                          |
|    TReDS Invoice Fraud Detection System                                       |
|    Hybrid Classical-Quantum Platform                                          |
|                                                                               |
|    Version: 1.0.0 MVP                                                         |
|                                                                               |
+===============================================================================+
    """
    print(banner)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def setup_logging():
    """Configure logging for the application."""
    from src.utils.logger import setup_logger
    logger = setup_logger("treds_fraud_detection")
    return logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="QGAI TReDS Invoice Fraud Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                      Run full pipeline with defaults
    python main.py --mode train         Train models only
    python main.py --mode predict       Run predictions only
    python main.py --verbose            Enable verbose output
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict", "full", "validate"],
        default="full",
        help="Execution mode (default: full)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom configuration file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output files"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--no-rings",
        action="store_true",
        help="Skip ring detection (classical only)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    return parser.parse_args()


def run_data_generation(config, verbose: bool = False):
    """
    Phase 2: Generate synthetic data.

    Args:
        config: Configuration object
        verbose: Enable verbose output

    Returns:
        Tuple of (entities_df, invoices_df)
    """
    print_section("PHASE 2: SYNTHETIC DATA GENERATION")

    from src.data_generation import EntityGenerator, InvoiceGenerator, DataValidator

    # Generate entities
    print("\n[1/3] Generating entities...")
    entity_gen = EntityGenerator(config.data_generation)
    entities_df = entity_gen.generate()
    print(f"      Generated {len(entities_df)} entities")
    print(f"      - Buyers: {len(entities_df[entities_df['entity_type'] == 'buyer'])}")
    print(f"      - Suppliers: {len(entities_df[entities_df['entity_type'] == 'supplier'])}")
    print(f"      - Ring Members: {len(entities_df[entities_df['is_ring_member']])}")

    # Generate invoices
    print("\n[2/3] Generating invoices...")
    invoice_gen = InvoiceGenerator(config.data_generation)
    invoices_df = invoice_gen.generate(entities_df)
    print(f"      Generated {len(invoices_df)} invoices")
    print(f"      - Legitimate: {len(invoices_df[~invoices_df['is_in_ring']])}")
    print(f"      - Ring-related: {len(invoices_df[invoices_df['is_in_ring']])}")

    # Validate data
    print("\n[3/3] Validating data...")
    validator = DataValidator()
    validation_result = validator.validate(entities_df, invoices_df)
    if validation_result.is_valid:
        print("      [OK] Data validation passed")
    else:
        print(f"      [FAIL] Data validation failed: {validation_result.errors}")

    return entities_df, invoices_df


def run_feature_engineering(entities_df, invoices_df, config, verbose: bool = False):
    """
    Phase 3: Feature engineering.

    Args:
        entities_df: Entity DataFrame
        invoices_df: Invoice DataFrame
        config: Configuration object
        verbose: Enable verbose output

    Returns:
        Tuple of (FeatureEngineeringResult, GraphBuildResult)
    """
    print_section("PHASE 3: FEATURE ENGINEERING")

    from src.feature_engineering import FeatureEngineer

    print("\n[1/1] Engineering features and building graph...")
    engineer = FeatureEngineer(config.features)
    feature_result = engineer.fit_transform(entities_df, invoices_df, build_graph=True)

    print(f"      Engineered {feature_result.n_features} features")
    print(f"      Samples: {feature_result.n_samples}")
    if feature_result.graph_result:
        print(f"      Graph: {feature_result.graph_result.n_nodes} nodes, {feature_result.graph_result.n_edges} edges")

    return feature_result, feature_result.graph_result


def run_classical_training(feature_result, config, verbose: bool = False):
    """
    Phase 4: Train classical default prediction model.

    Args:
        feature_result: FeatureEngineeringResult from Phase 3
        config: Configuration object
        verbose: Enable verbose output

    Returns:
        Tuple of (ModelTrainer result, DefaultPredictor, feature matrix X, target y)
    """
    print_section("PHASE 4: CLASSICAL DEFAULT PREDICTION")

    from src.classical import ModelTrainer
    from src.feature_engineering import FeatureEngineer

    # Extract feature matrix and target from feature engineering result
    engineer = FeatureEngineer(config.features)
    X, y = engineer.get_feature_matrix(feature_result)

    print(f"\n      Feature matrix: {X.shape}")
    print(f"      Default rate: {y.mean():.2%}")

    print("\n[1/2] Training Random Forest model...")
    trainer = ModelTrainer(config.classical)
    training_result = trainer.train(X, y, feature_result.feature_names)

    print(f"\n[2/2] Model Performance:")
    print(f"      - AUC-ROC: {training_result.test_evaluation.auc_roc:.4f} (target: >={config.classical.target_auc_roc})")
    print(f"      - Average Precision: {training_result.test_evaluation.average_precision:.4f}")
    print(f"      - F1-Score: {training_result.test_evaluation.f1:.4f}")
    if training_result.training_result.evaluation.cv_mean:
        print(f"      - CV AUC-ROC: {training_result.training_result.evaluation.cv_mean:.4f}")

    # Check target
    if training_result.test_evaluation.auc_roc >= config.classical.target_auc_roc:
        print("      [OK] AUC-ROC target met!")
    else:
        print("      [WARNING] AUC-ROC below target")

    predictor = training_result.model
    return training_result, predictor, X, y


def run_quantum_detection(graph_result, invoices_df, config, verbose: bool = False):
    """
    Phase 5: QUBO-based ring detection.

    Args:
        graph_result: GraphBuildResult containing transaction graph
        invoices_df: Invoice DataFrame (for ground truth)
        config: Configuration object
        verbose: Enable verbose output

    Returns:
        Ring detection result with community assignments
    """
    print_section("PHASE 5: QUANTUM RING DETECTION (QUBO)")

    from src.quantum import RingDetector

    print("\n[1/2] Running QUBO-based community detection...")
    detector = RingDetector(config=config.quantum)

    # Pass the modularity matrix, node list, and adjacency matrix from graph_result
    result = detector.detect(
        modularity_matrix=graph_result.modularity_matrix,
        node_list=graph_result.node_list,
        adjacency_matrix=graph_result.adjacency_matrix
    )

    n_variables = len(graph_result.node_list) * detector.k_communities
    print(f"      QUBO size: {n_variables} variables")
    print(f"      Communities found: {result.n_communities_found}")
    print(f"      Modularity score: {result.modularity:.4f}")

    print("\n[2/2] Evaluating ring detection...")
    # Calculate ring recovery rate against ground truth
    ground_truth_rings = invoices_df[invoices_df['is_in_ring']]['ring_id'].unique()
    high_risk_communities = [c for c in result.communities if c.ring_score >= config.quantum.ring_detection_threshold]

    print(f"\n      Ring Detection Results:")
    print(f"      - Communities detected: {result.n_communities_found}")
    print(f"      - High-risk communities: {len(high_risk_communities)}")
    print(f"      - Ground truth rings: {len(ground_truth_rings)}")

    # Build ring_scores dict mapping entity_id -> ring_score
    ring_scores = detector.get_ring_scores(result)

    return result, ring_scores


def run_pipeline_integration(predictor, ring_scores, X, invoices_df, config, verbose: bool = False):
    """
    Phase 6: Integrate classical and quantum components.

    Args:
        predictor: Trained default predictor
        ring_scores: Ring probability scores (entity_id -> score)
        X: Feature matrix for default prediction
        invoices_df: Invoice DataFrame
        config: Configuration object
        verbose: Enable verbose output

    Returns:
        Tuple of (RiskScoringResult, targeting DataFrame)
    """
    print_section("PHASE 6: HYBRID PIPELINE INTEGRATION")

    from src.pipeline import RiskScorer

    print("\n[1/3] Computing default probabilities...")
    default_probs = predictor.predict_proba(X)
    print(f"      Mean P(default): {default_probs.mean():.4f}")

    print("\n[2/3] Computing composite risk scores...")
    risk_scorer = RiskScorer(config=config.risk_scoring)
    risk_result = risk_scorer.score(invoices_df, default_probs, ring_scores)

    summary = risk_scorer.get_risk_summary(risk_result)

    print(f"\n      Risk Distribution:")
    for level, count in summary.get('risk_level_counts', {}).items():
        pct = count / summary['total_invoices'] * 100
        print(f"      - {level}: {count} ({pct:.1f}%)")

    print(f"\n      High risk: {summary['n_high_risk']} ({summary['pct_high_risk']:.2f}%)")
    print(f"      Ring associated: {summary['n_ring_associated']} ({summary['pct_ring_associated']:.2f}%)")

    print("\n[3/3] Generating targeting list...")
    targeting_list = risk_scorer.get_targeting_list(risk_result, top_n=20)
    print(f"      Top 20 invoices by composite score generated")

    return risk_result, targeting_list


def run_explainability(predictor, X, feature_names, config, verbose: bool = False):
    """
    Phase 7: Generate explanations.

    Args:
        predictor: Trained DefaultPredictor model
        X: Feature matrix
        feature_names: List of feature names
        config: Configuration object
        verbose: Enable verbose output

    Returns:
        ExplanationResult
    """
    print_section("PHASE 7: EXPLAINABILITY FRAMEWORK")

    from src.explainability import SHAPExplainer

    print("\n[1/2] Computing SHAP values...")
    shap_explainer = SHAPExplainer(config.explainability)
    shap_explainer.fit(predictor.model, feature_names)

    n_samples = min(100, len(X))
    explanation_result = shap_explainer.explain(X, n_samples=n_samples)

    print(f"      Samples explained: {explanation_result.n_samples}")
    print(f"      Expected value: {explanation_result.expected_value:.4f}")

    print(f"\n[2/2] Feature importance (top 5):")
    for feat, imp in explanation_result.global_explanation.feature_ranking[:5]:
        direction = '+' if explanation_result.global_explanation.mean_shap_values[feat] > 0 else '-'
        print(f"      - {feat}: {imp:.4f} ({direction})")

    print("\n      [OK] Explanations generated")

    return explanation_result


def run_validation(predictions, invoices_df, config, verbose: bool = False):
    """
    Phase 8: Validate against success criteria.

    Args:
        predictions: Final predictions
        invoices_df: Ground truth data
        config: Configuration object
        verbose: Enable verbose output

    Returns:
        Validation results dictionary
    """
    print_section("PHASE 8: VALIDATION & SUCCESS CRITERIA")

    from config.constants import SuccessCriteria

    results = {}

    print("\n      MVP Success Criteria Check:")
    print("      " + "-" * 50)

    # This will be fully implemented in Phase 8
    criteria = [
        ("Default Model AUC-ROC", ">=0.75", "PENDING"),
        ("Ring Detection Modularity", ">=0.30", "PENDING"),
        ("Ring Recovery Rate", ">=70%", "PENDING"),
        ("Explainability Coverage", "100%", "PENDING"),
        ("Quantum Readiness", "Full", "PENDING"),
    ]

    for name, target, status in criteria:
        print(f"      {name}: {target} [{status}]")

    return results


def save_outputs(predictions, targeting_list, config, output_dir: Optional[Path] = None):
    """
    Save all outputs to files.

    Args:
        predictions: Predictions DataFrame
        targeting_list: Prioritized targeting list
        config: Configuration object
        output_dir: Output directory path
    """
    print_section("SAVING OUTPUTS")

    if output_dir is None:
        output_dir = config.paths.output_data_dir

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save predictions
    pred_file = output_dir / f"predictions_{timestamp}.csv"
    predictions.to_csv(pred_file, index=False)
    print(f"      Predictions saved: {pred_file}")

    # Save targeting list
    target_file = output_dir / f"targeting_list_{timestamp}.csv"
    targeting_list.to_csv(target_file, index=False)
    print(f"      Targeting list saved: {target_file}")

    print(f"\n      [OK] All outputs saved to {output_dir}")


def main():
    """Main entry point."""
    start_time = time.time()

    # Parse arguments
    args = parse_arguments()

    # Print banner
    print_banner()

    print(f"Execution Mode: {args.mode.upper()}")
    print(f"Random Seed: {args.seed}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load configuration
    from config import get_config
    config = get_config()

    try:
        if args.mode in ["full", "train"]:
            # Phase 2: Data Generation
            entities_df, invoices_df = run_data_generation(config, args.verbose)

            # Phase 3: Feature Engineering
            feature_result, graph = run_feature_engineering(
                entities_df, invoices_df, config, args.verbose
            )

            # Phase 4: Classical Training
            training_result, predictor, X, y = run_classical_training(
                feature_result, config, args.verbose
            )

            if not args.no_rings:
                # Phase 5: Quantum Detection
                detector, ring_scores = run_quantum_detection(
                    graph, invoices_df, config, args.verbose
                )
            else:
                ring_scores = {}

            if args.mode == "full":
                # Phase 6: Integration
                risk_result, targeting_list = run_pipeline_integration(
                    predictor, ring_scores, X, invoices_df, config, args.verbose
                )

                # Phase 7: Explainability
                explanation_result = run_explainability(
                    predictor, X, feature_result.feature_names, config, args.verbose
                )

                # Phase 8: Validation
                validation_results = run_validation(
                    risk_result.scores_df, invoices_df, config, args.verbose
                )

                # Save outputs
                save_outputs(risk_result.scores_df, targeting_list, config, args.output_dir)

        elif args.mode == "predict":
            print("\nPredict mode requires trained models. Please run with --mode train first.")
            sys.exit(1)

        elif args.mode == "validate":
            print("\nValidation mode - checking system components...")
            # Run validation checks
            run_validation(None, None, config, args.verbose)

        # Print execution summary
        elapsed_time = time.time() - start_time
        print_section("EXECUTION COMPLETE")
        print(f"\n      Total execution time: {elapsed_time:.2f} seconds")
        print(f"      Status: SUCCESS")

    except Exception as e:
        print(f"\n      ERROR: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
