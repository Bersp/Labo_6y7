/* #include <Wire.h> */
/* #include <Adafruit_Sensor.h> */
/* #include <Adafruit_ADXL345_U.h> */
/* Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(); */
/* void setup(void) */
/* { */
   /* Serial.begin(9600); */
   /* if(!accel.begin()) */
   /* { */
      /* Serial.println("No valid sensor found"); */
   /* } */
  /* Serial.println("t, x, y, z"); */
/* } */
/* void loop(void) */
/* { */
   /* sensors_event_t event; */
   /* accel.getEvent(&event); */
   /* Serial.print(millis()); */
   /* Serial.print(", "); */
   /* Serial.print(event.acceleration.x); */
   /* Serial.print(", "); */
   /* Serial.print(event.acceleration.y); */
   /* Serial.print(", "); */
   /* Serial.println(event.acceleration.z); */
/* } */

#include <Wire.h>
#include <SparkFun_ADXL345.h>
ADXL345 adxl = ADXL345();

void setup()
{
	Serial.begin(115200);
	Serial.println("START");
	Serial.println();

	adxl.powerOn();
	adxl.setRangeSetting(2);       //Definir el rango, valores 2, 4, 8 o 16
	adxl.set_bw(ADXL345_BW_1600);
}

void loop()
{
	// Leer los valores e imprimirlos
	int x, y, z;
	adxl.readAccel(&x, &y, &z);
	Serial.print(millis());
	Serial.print(",");
	Serial.print(x);
	Serial.print(",");
	Serial.print(y);
	Serial.print(",");
	Serial.println(z);
}


