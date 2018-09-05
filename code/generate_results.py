###############################################################################
from utils import load_predictions, generate_metrics, save_metrics_summary
from utils import summarize_metrics_folds, summarize_metrics_pool


if __name__ == "__main__":

	print "Loading data"
	predictions_dict = load_predictions()

	print "Started analysis"
	metrics_dict = generate_metrics(predictions_dict)
	print "Generated metrics per fold"
	summary_folds = summarize_metrics_folds(metrics_dict)
	print "Generated summary of all folds"
	summary_pool = summarize_metrics_pool(summary_folds)
	print "Generated pool summary"

	print "Finished analysis"
	save_metrics_summary(summary_pool, "pool_comparison")

	print "Stored metrics"