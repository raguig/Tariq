package com.tariq.util;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.Producer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.serialization.StringSerializer;

import java.util.Properties;
import java.util.concurrent.ExecutionException;

public class KafkaProducerUtil {
    private static final ObjectMapper objectMapper = new ObjectMapper();
    private static Producer<String, String> producer;

    public static void initialize(String bootstrapServers) {
        Properties props = new Properties();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        props.put(ProducerConfig.ACKS_CONFIG, "1");

        producer = new KafkaProducer<>(props);
        System.out.println("[INFO] Kafka producer initialized with bootstrap servers: " + bootstrapServers);
    }

    public static void sendMessage(String topic, String key, Object value) {
        try {
            String jsonValue = objectMapper.writeValueAsString(value);
            ProducerRecord<String, String> record = new ProducerRecord<>(topic, key, jsonValue);

            producer.send(record, (metadata, exception) -> {
                if (exception != null) {
                    System.err.println("[ERROR] Error sending message to topic " + topic + ": " + exception.getMessage());
                } else {
                    System.out.println("[DEBUG] Message sent to topic " + metadata.topic() +
                            ", partition " + metadata.partition() +
                            ", offset " + metadata.offset());
                }
            }).get(); // Optional sync
        } catch (Exception e) {
            System.err.println("[ERROR] Error serializing or sending message: " + e.getMessage());
        }
    }

    public static void closeProducer() {
        if (producer != null) {
            producer.flush();
            producer.close();
            System.out.println("[INFO] Kafka producer closed");
        }
    }
}
