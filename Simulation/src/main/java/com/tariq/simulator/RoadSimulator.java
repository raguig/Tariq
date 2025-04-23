package com.tariq.simulator;

import com.tariq.model.Road;
import com.tariq.model.TrafficData;
import com.tariq.model.Vehicle;
import com.tariq.util.KafkaProducerUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;


public class RoadSimulator implements Runnable{
    private final Road road;
    private final Random random=new Random();
    private final AtomicBoolean running=new AtomicBoolean(false);
    private final List<Vehicle> activeVehicles=new ArrayList<>();
    private final int simulationSpeed; // milliseconds between updates
    private double averageSpeed = 60.0; // initial average speed in km/h
    private static final Logger logger = LoggerFactory.getLogger(RoadSimulator.class);

    public RoadSimulator(Road road, int simulationSpeed) {
        this.road = road;
        this.simulationSpeed = simulationSpeed;
    }

    @Override
    public void run(){
        running.set(true);
        while(running.get()){
            try{
                simulateTrafficByTimeOfDay();

                updateAverageSpeed();

                TrafficData data = new TrafficData(
                  road.getId(),
                  road.getCurrentVehicleCount(),
                  road.getCongestionLevel(),
                  averageSpeed
                );

                KafkaProducerUtil.sendMessage("traffic-data",road.getId(),data);
                logger.info("Road '{}': {} vehicles, congestion level {}, avg speed {:.2f} km/h",
                         road.getCurrentVehicleCount(), road.getCongestionLevel(), averageSpeed);

                Thread.sleep(simulationSpeed);
            }catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                running.set(false);
            }
        }
    }


    private void simulateTrafficByTimeOfDay() {
        LocalDateTime now = LocalDateTime.now();
        int hour = now.getHour();
        int dayOfWeek = now.getDayOfWeek().getValue();
        boolean isWeekend = dayOfWeek >= 6;

        // Reset vehicle count for more realistic daily patterns
        if (hour == 3) { // 3 AM - minimum traffic
            road.setVehicleCount((int) (road.getMaxCapacity() * 0.05));
        }

        double trafficFactor;

        // Weekday patterns
        if (!isWeekend) {
            if (hour >= 7 && hour <= 9) {
                // Morning rush hour
                trafficFactor = 0.7 + (random.nextDouble() * 0.2);
            } else if (hour >= 16 && hour <= 19) {
                // Evening rush hour
                trafficFactor = 0.65 + (random.nextDouble() * 0.25);
            } else if (hour >= 10 && hour <= 15) {
                // Midday
                trafficFactor = 0.4 + (random.nextDouble() * 0.15);
            } else if (hour >= 20 && hour <= 23) {
                // Evening
                trafficFactor = 0.25 + (random.nextDouble() * 0.15);
            } else {
                // Night
                trafficFactor = 0.1 + (random.nextDouble() * 0.1);
            }
        } else {
            // Weekend patterns
            if (hour >= 10 && hour <= 16) {
                // Daytime weekend
                trafficFactor = 0.45 + (random.nextDouble() * 0.25);
            } else if (hour >= 17 && hour <= 22) {
                // Evening weekend
                trafficFactor = 0.35 + (random.nextDouble() * 0.2);
            } else {
                // Night weekend
                trafficFactor = 0.15 + (random.nextDouble() * 0.1);
            }
        }

        // Calculate target vehicle count
        int targetCount = (int) (road.getMaxCapacity() * trafficFactor);
        int currentCount = road.getCurrentVehicleCount();

        // Gradually adjust to target (avoid abrupt changes)
        if (currentCount < targetCount) {
            // Add vehicles
            int addCount = Math.min(10, targetCount - currentCount);
            for (int i = 0; i < addCount; i++) {
                road.addVehicle();
            }
        } else if (currentCount > targetCount) {
            // Remove vehicles
            int removeCount = Math.min(10, currentCount - targetCount);
            for (int i = 0; i < removeCount; i++) {
                road.removeVehicle();
            }
        }
    }

    private void updateAverageSpeed() {
        // Adjust average speed based on congestion level
        int congestionLevel = road.getCongestionLevel();
        double baseSpeed = 60.0; // base speed in km/h

        switch (congestionLevel) {
            case 0: // Free flow
                averageSpeed = baseSpeed + (random.nextDouble() * 20);
                break;
            case 1: // Light
                averageSpeed = baseSpeed - 5 + (random.nextDouble() * 10);
                break;
            case 2: // Moderate
                averageSpeed = baseSpeed * 0.7 + (random.nextDouble() * 10);
                break;
            case 3: // Heavy
                averageSpeed = baseSpeed * 0.5 + (random.nextDouble() * 8);
                break;
            case 4: // Very Heavy
                averageSpeed = baseSpeed * 0.3 + (random.nextDouble() * 5);
                break;
            case 5: // Gridlock
                averageSpeed = 5 + (random.nextDouble() * 10);
                break;
            default:
                averageSpeed = baseSpeed;
        }
    }





    public void stop() {
        running.set(false);
    }
}
