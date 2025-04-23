package com.tariq.simulator;


import com.tariq.model.Road;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class TrafficSimulationManager {
    private static final Logger logger = LoggerFactory.getLogger(TrafficSimulationManager.class);
    private final List<Road> roads = new ArrayList<>();
    private final List<RoadSimulator> simulators = new ArrayList<>();
    private ExecutorService executorService;

    public void addRoad(Road road) {
        roads.add(road);
        RoadSimulator simulator = new RoadSimulator(road, 10000); // Update every 1 second
        simulators.add(simulator);
    }

    public void startSimulation() {
        int threadCount = Math.min(roads.size(), Runtime.getRuntime().availableProcessors());
        executorService = Executors.newFixedThreadPool(threadCount);

        for (RoadSimulator simulator : simulators) {
            executorService.submit(simulator);
        }

        logger.info("Started traffic simulation with {} roads on {} threads", roads.size(), threadCount);
    }

    public void stopSimulation() {
        for (RoadSimulator simulator : simulators) {
            simulator.stop();
        }

        if (executorService != null) {
            executorService.shutdown();
        }

        logger.info("Stopped traffic simulation");
    }

    public List<Road> getRoads() {
        return new ArrayList<>(roads);
    }
}