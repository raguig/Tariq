package com.tariq.model;


import java.util.UUID;
public class Vehicle {
    private final String id;
    private final double speed;
    private final long timestamp;

    public Vehicle(double speed){
        this.id=UUID.randomUUID().toString();
        this.speed=speed;
        this.timestamp=System.currentTimeMillis();
    }

    public String getId() {
        return id;
    }
    public double getSpeed() {
        return speed;
    }
    public long getTimestamp() {
        return timestamp;
    }

    @Override
    public String toString() {
        return "Vehicle{" +
                "id='" + id + '\'' +
                ", speed=" + speed +
                ", timestamp=" + timestamp +
                '}';
    }


}
