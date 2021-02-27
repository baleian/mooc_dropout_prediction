package beom.kafka.exam.partitioner;

import org.apache.kafka.clients.producer.Partitioner;
import org.apache.kafka.common.Cluster;

import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class RoundRobinPartitioner implements Partitioner {

    private AtomicInteger n = new AtomicInteger(0);

    public int partition(String topic, Object key, byte[] keyBytes, Object value, byte[] valueBytes, Cluster cluster) {
        int k = Integer.parseInt((String) key);
        if (k <= 33) {
            return 0;
        }
        if (k <= 66) {
            return 1;
        }
        return 2;
    }

    public void close() {

    }

    public void configure(Map<String, ?> map) {

    }

}
