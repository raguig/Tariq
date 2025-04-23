package org.example;


import com.tariq.model.Road;
import com.tariq.simulator.TrafficSimulationManager;
import com.tariq.util.KafkaProducerUtil;

import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        KafkaProducerUtil.initialize("localhost:9092");

        TrafficSimulationManager simulationManager = new TrafficSimulationManager();

        simulationManager.addRoad(new Road("road-1"));
        simulationManager.addRoad(new Road("road-2"));
        simulationManager.addRoad(new Road("road-3" ));
        simulationManager.addRoad(new Road("road-4"));

        System.out.println("[INFO] Starting Tariq Traffic Management System");
        simulationManager.startSimulation();

        System.out.println("[INFO] Traffic simulation is running. Press ENTER to stop.");
        try (Scanner scanner = new Scanner(System.in)) {
            scanner.nextLine();
        }


        simulationManager.stopSimulation();
        KafkaProducerUtil.closeProducer();
        System.out.println("[INFO] Tariq Traffic Management System stopped");
    }
}
