package beom.kafka.exam.wordcount;

import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.KeyValue;
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.StreamsConfig;
import org.apache.kafka.streams.kstream.Grouped;

import java.util.Arrays;
import java.util.Properties;

public class WordCountApp {

    public static void main(String[] args) {
        Properties properties = new Properties();
        properties.put(StreamsConfig.APPLICATION_ID_CONFIG, "streams-wordcount-application");
        properties.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        properties.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        properties.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        properties.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");

        StreamsBuilder builder = new StreamsBuilder();
        builder.<String, String>stream("streams-wordcount-input")
                .flatMapValues(value -> Arrays.asList(value.split(" ")))
                .selectKey((key, value) -> value.toUpperCase())
                .groupByKey()
                .count()
                .mapValues(value -> Long.toString(value))
                .toStream()
                .to(WordCountConsumer.CONSUME_TOPIC)
//                .foreach((key, value) -> System.out.println(key + ":" +  value))
        ;

        KafkaStreams streams = new KafkaStreams(builder.build(), properties);
        streams.start();
    }

}
