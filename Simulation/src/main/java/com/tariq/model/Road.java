package com.tariq.model;

import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;


public class Road {

    private final String id;
    private final int maxCapacity=250;
    private final AtomicInteger currentVehicleCount = new AtomicInteger(0);
    private final AtomicInteger congestionLevel = new AtomicInteger(0);
    private final Random random = new Random();

    public Road(String id){
        this.id=id;
    }

    public String getId() {
        return id;
    }
    public int getMaxCapacity() {
        return maxCapacity;
    }
    public int getCurrentVehicleCount() {
        return currentVehicleCount.get();
    }

    public void  addVehicle(){
        int increment = random.nextInt(20)+1;
        int newCount=currentVehicleCount.addAndGet(increment);
        if(newCount>maxCapacity){
            currentVehicleCount.set(maxCapacity);
        }
        updateCongestionLevel();
    }

    public void removeVehicle(){
        int decrement = random.nextInt(20)+1;
        int newCount=currentVehicleCount.addAndGet(-decrement);
        if(newCount<0){
            currentVehicleCount.set(0);
        }
        updateCongestionLevel();
    }

    public void setVehicleCount(int count){
        currentVehicleCount.set(Math.max(0,count));
        updateCongestionLevel();
    }

    public int getCongestionLevel() {
        return congestionLevel.get();
    }

    private synchronized void updateCongestionLevel() {
        double ratio = (double) currentVehicleCount.get() / maxCapacity;
        double roundedRatio = Math.floor(ratio * 100) / 100.0;
        ratio = roundedRatio;


        if (roundedRatio < 0.20) {
            congestionLevel.set(0);
        } else if (ratio < 0.40) {
            congestionLevel.set(1);
        } else if (ratio < 0.60) {
            congestionLevel.set(2);
        } else if (ratio < 0.80) {
            congestionLevel.set(3);
        } else if (ratio < 0.90) {
            congestionLevel.set(4);
        } else {
            congestionLevel.set(5);
        }
        System.out.println("Current vehicle count: " + currentVehicleCount.get() + " / Max capacity: " + maxCapacity+" Ratio: " + ratio+" congestion:"+getCongestionLevel()+" result: "+(ratio<0.4));

    }

    @Override
    public String toString() {
        return "Road{" +
                "id='" + id + '\'' +
                ", currentVehicleCount=" + currentVehicleCount +
                ", congestionLevel=" + congestionLevel +
                '}';
    }



}
