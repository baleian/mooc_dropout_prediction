package beom.kafka.exam.user_enrich;

import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.StreamsConfig;
import org.apache.kafka.streams.kstream.*;

import java.util.Properties;

public class UserEventEnricherApp {
    public static void main(String[] args) {

        Properties config = new Properties();
        config.put(StreamsConfig.APPLICATION_ID_CONFIG, "user-event-enricher-app");
        config.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        config.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
        config.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        config.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());

        StreamsBuilder builder = new StreamsBuilder();

        // we get a global table out of Kafka. This table will be replicated on each Kafka Streams application
        // the key of our globalKTable is the user ID
        GlobalKTable<String, String> usersGlobalTable = builder.globalTable("user-table");

        // we get a stream of user purchases
        KStream<String, String> userPurchases = builder.stream("user-purchases");

        // we want to enrich that stream
        userPurchases
                .join(
                        usersGlobalTable,
                        (key, value) -> key,
                        (leftValue, rightValue) -> leftValue + ", " + rightValue
                )
                .foreach((key, value) -> System.out.println("[JOIN] " + key + " --- " + value))
                ;

        // we want to enrich that stream using a Left Join
        userPurchases
                .leftJoin(
                        usersGlobalTable,
                        (key, value) -> key,
                        (leftValue, rightValue) -> leftValue + ", " + rightValue
                )
                .foreach((key, value) -> System.out.println("[LEFT-JOIN] " + key + " --- " + value))
        ;

        KafkaStreams streams = new KafkaStreams(builder.build(), config);
        streams.cleanUp(); // only do this in dev - not in prod
        streams.start();

        // print the topology
        System.out.println(streams.toString());

        // shutdown hook to correctly close the streams application
        Runtime.getRuntime().addShutdownHook(new Thread(streams::close));

    }
}
