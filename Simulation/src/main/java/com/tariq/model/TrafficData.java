package com.tariq.model;


import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class TrafficData {
    private final String roadId;
    private final String timestamp;
    private final int vehicleCount;
    private final int congestionLevel;
    private final double averageSpeed;
    private final int hour;
    private final int dayOfWeek;
    private final boolean isWeekend;

    public TrafficData(String roadId,int vehicleCount,int congestionLevel,double averageSpeed){
        this.roadId=roadId;
        this.vehicleCount=vehicleCount;
        this.congestionLevel=congestionLevel;
        this.averageSpeed=averageSpeed;
        LocalDateTime now=LocalDateTime.now();

        this.timestamp=now.format(DateTimeFormatter.ISO_DATE_TIME);
        this.hour=now.getHour();
        this.dayOfWeek=now.getDayOfWeek().getValue();
        this.isWeekend =(dayOfWeek==6 || dayOfWeek==7);

    }

    public String getRoadId() {
        return roadId;
    }

    public String getTimestamp() {
        return timestamp;
    }

    public int getVehicleCount() {
        return vehicleCount;
    }

    public int getCongestionLevel() {
        return congestionLevel;
    }

    public double getAverageSpeed() {
        return averageSpeed;
    }

    public int getHour() {
        return hour;
    }

    public int getDayOfWeek() {
        return dayOfWeek;
    }

    public boolean getIsWeekend() {
        return isWeekend;
    }

    @Override
    public String toString() {
        return "TrafficData{" +
                "roadId='" + roadId + '\'' +
                ", timestamp='" + timestamp + '\'' +
                ", vehicleCount=" + vehicleCount +
                ", congestionLevel=" + congestionLevel +
                ", averageSpeed=" + averageSpeed +
                ", hour=" + hour +
                ", dayOfWeek=" + dayOfWeek +
                ", isWeekend=" + isWeekend +
                '}';
    }


}
