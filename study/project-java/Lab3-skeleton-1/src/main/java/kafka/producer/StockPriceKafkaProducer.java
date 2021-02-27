package kafka.producer;

import kafka.StockAppConstants;
import kafka.producer.model.StockPrice;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.Producer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.common.serialization.StringSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class StockPriceKafkaProducer {
    private static final Logger logger = LoggerFactory.getLogger(StockPriceKafkaProducer.class);

    private static Producer<String, StockPrice> createProducer() {
        final Properties props = new Properties();
        setupBootstrapAndSerializers(props);
        return new KafkaProducer<>(props);
    }

    private static void setupBootstrapAndSerializers(Properties props) {
        // 1. Set your ProducerConfig.BOOTSTRAP_SERVERS_CONFIG
        // 2. Set your ProducerConfig.CLIENT_ID_CONFIG
        // 3. Set your key serializer
        // 4. Set your value serializer
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, StockAppConstants.BOOTSTRAP_SERVERS);
        props.put(ProducerConfig.CLIENT_ID_CONFIG, "StockPriceKafkaProducer");
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StockPriceSerializer.class.getName());
//        props.put(ProducerConfig.ACKS_CONFIG, "all");
    }


    public static void main(String... args)
            throws Exception {
        //Create Kafka Producer
        final Producer<String, StockPrice> producer = createProducer();
        //Create StockSender list
        final List<StockSender> stockSenders = getStockSenderList(producer);

        //Create a thread pool so every stock sender gets it own.
        // Increase by 1 to fit metrics.
        final ExecutorService executorService =
                Executors.newFixedThreadPool(stockSenders.size() + 1);

        executorService.submit(new MetricsProducerReporter(producer));

        //Run each stock sender in its own thread.
        stockSenders.forEach(executorService::submit);
    }

    private static List<StockSender> getStockSenderList(
            final Producer<String, StockPrice> producer) {
        return Arrays.asList(
                new StockSender(StockAppConstants.TOPIC,
                        new StockPrice("IBM", 100, 99),
                        new StockPrice("IBM", 50, 10),
                        producer,
                        1, 10
                ),
                new StockSender(StockAppConstants.TOPIC,
                        new StockPrice("SUN", 100, 99),
                        new StockPrice("SUN", 50, 10),
                        producer,
                        1, 10
                ),
                new StockSender(StockAppConstants.TOPIC,
                        new StockPrice("GOOG", 500, 99),
                        new StockPrice("GOOG", 400, 10),
                        producer,
                        1, 10
                ),
                new StockSender(StockAppConstants.TOPIC,
                        new StockPrice("INTEL", 100, 99),
                        new StockPrice("INTEL", 50, 10),
                        producer,
                        1, 10
                )
        );
    }

}