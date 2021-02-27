package kafka.producer;

import kafka.producer.model.StockPrice;
import org.apache.kafka.clients.producer.Producer;
import org.apache.kafka.common.Metric;
import org.apache.kafka.common.MetricName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

public class MetricsProducerReporter implements Runnable {
    private final Producer<String, StockPrice> producer;
    private static final Logger logger = LoggerFactory.getLogger(MetricsProducerReporter.class);

    public MetricsProducerReporter(
            final Producer<String, StockPrice> producer) {
        this.producer = producer;
    }

    @Override
    public void run() {
        while (true) {
            // 1. Get metrics from your producer
            // 2. Display the metrics
            final Map<MetricName, ? extends Metric> metrics = producer.metrics();

//            metrics.forEach((metricName, metric) -> logger.info(String.format(
//                    "\nMetric\t %s, \t %s, \t %s, \n\t\t%s\n",
//                    metricName.group(), metricName.name(), metric.metricValue(), metricName.description()
//            )));
            displayMetrics(metrics);
            try {
                Thread.sleep(3_000);
            } catch (InterruptedException e) {
                //logger.warn("metrics interrupted");
                Thread.interrupted();
                break;
            }
        }
    }

    //Used to Filter just the metrics we want
    private final Set<String> metricsNameFilter = new HashSet<>(Arrays.asList(
            "record-queue-time-avg", "record-send-rate", "records-per-request-avg",
            "request-size-max", "network-io-rate", "record-queue-time-avg",
            "incoming-byte-rate", "batch-size-avg", "response-rate", "requests-in-flight"
    ));

    static class MetricPair {
        private final MetricName metricName;
        private final Metric metric;
        MetricPair(MetricName metricName, Metric metric) {
            this.metricName = metricName;
            this.metric = metric;
        }
        public String toString() {
            return metricName.group() + "." + metricName.name();
        }
    }

    private void displayMetrics(Map<MetricName, ? extends Metric> metrics) {
        final Map<String, MetricPair> metricsDisplayMap = metrics.entrySet().stream()
                .filter(metricNameEntry ->
                        metricsNameFilter.contains(metricNameEntry.getKey().name()))
                .filter(metricNameEntry ->
                        !Double.isInfinite((double) metricNameEntry.getValue().metricValue()) &&
                                !Double.isNaN((double) metricNameEntry.getValue().metricValue()) &&
                                (double) metricNameEntry.getValue().metricValue() != 0
                )
                .map(entry -> new MetricPair(entry.getKey(), entry.getValue()))
                .collect(Collectors.toMap(
                        MetricPair::toString, it -> it, (a, b) -> a, TreeMap::new
                ));

        //Output metrics
        final StringBuilder builder = new StringBuilder(255);
        builder.append("\n---------------------------------------\n");
        metricsDisplayMap.entrySet().forEach(entry -> {
            MetricPair metricPair = entry.getValue();
            String name = entry.getKey();
            builder.append(String.format(Locale.US, "%50s%25s\t\t%,-10.2f\t\t%s\n",
                    name,
                    metricPair.metricName.name(),
                    metricPair.metric.metricValue(),
                    metricPair.metricName.description()));
        });
        builder.append("\n---------------------------------------\n");
        logger.info(builder.toString());
    }


}
