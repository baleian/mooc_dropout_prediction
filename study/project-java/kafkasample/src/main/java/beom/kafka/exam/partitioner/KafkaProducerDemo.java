package beom.kafka.exam.partitioner;

import org.apache.kafka.clients.producer.*;
import org.apache.kafka.common.serialization.StringSerializer;

import java.util.Properties;
import java.util.concurrent.CountDownLatch;

public class KafkaProducerDemo {

    public static void main(String[] args) throws InterruptedException {
        final Properties properties = new Properties();
        properties.setProperty("bootstrap.servers", "localhost:9092");
        properties.setProperty("key.serializer", StringSerializer.class.getName());
        properties.setProperty("value.serializer", StringSerializer.class.getName());
        properties.setProperty("acks", "1");
        properties.setProperty("retries", "3");
        properties.setProperty("linger.ms", "1");
        properties.setProperty("partitioner.class", RoundRobinPartitioner.class.getName());

        final CountDownLatch latch = new CountDownLatch(10);

        for (int i = 0; i < 10; i++) {
            final int id = i;
            new Thread(new Runnable() {
                public void run() {
                    Producer<String, String> producer = new KafkaProducer<String, String>(properties);
                    for (int key = id*10; key < (id+1)*10; key++) {
                        final int kk = key;
                        producer.send(
                                new ProducerRecord<String, String>("first_topic", Integer.toString(key), "message " + key),
                                new Callback() {
                                    public void onCompletion(RecordMetadata recordMetadata, Exception e) {
                                        System.out.println("sent message " + kk + " " + recordMetadata.offset());
                                    }
                                }
                        );
                        System.out.println("order message " + kk);
                    }
                    producer.close();
                    latch.countDown();
                }
            }).start();
        }

        latch.await();
        System.out.println("All threads terminated.");
    }

}
