package beom.kafka.stream.exam;

import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.StreamsConfig;
import org.apache.kafka.streams.kstream.Joined;
import org.apache.kafka.streams.kstream.KStream;
import org.apache.kafka.streams.kstream.KTable;

import java.util.Properties;

public class Practice1 {

    public static void main(String[] args) {
        Properties properties = new Properties();
        properties.put(StreamsConfig.APPLICATION_ID_CONFIG, "application-practice15");
        properties.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        properties.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
        properties.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        properties.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());

        StreamsBuilder builder = new StreamsBuilder();

        KStream<String, String> kStream = builder.stream("join-stream");
        KTable<String, String> kTable = builder.table("join-table");

        kStream
                .join(kTable, (leftValue, rightValue) -> leftValue + "," + rightValue,
                        Joined.with(Serdes.String(), Serdes.String(), Serdes.String()))
                .foreach((key, value) -> System.out.println(key + ":" + value))
        ;

        KafkaStreams streams = new KafkaStreams(builder.build(), properties);
        streams.start();
    }

}
