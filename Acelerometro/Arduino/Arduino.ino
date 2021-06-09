//////////////
// ADAFRUIT //
//////////////

/* #include <Wire.h> */
/* #include <Adafruit_Sensor.h> */
/* #include <Adafruit_ADXL345_U.h> */
/* Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(); */
/* void setup(void) */
/* { */
	/* Serial.begin(115200); */
	/* if(!accel.begin()) */
	/* { */
		/* Serial.println("No valid sensor found"); */
	/* } */
	/* Serial.println("START"); */
	/* accel.setRange(ADXL345_RANGE_4_G); */
/* } */
/* void loop(void) */
/* { */
	/* sensors_event_t event; */
	/* accel.getEvent(&event); */
	/* Serial.print(millis()); */
	/* Serial.print(","); */
	/* Serial.print(event.acceleration.x); */
	/* Serial.print(","); */
	/* Serial.print(event.acceleration.y); */
	/* Serial.print(","); */
	/* Serial.println(event.acceleration.z); */
/* } */

//////////////
// SPARKFUN //
//////////////

#include <Wire.h>
#include <SparkFun_ADXL345.h>
ADXL345 adxl = ADXL345();

void setup()
{
	Serial.begin(115200);
	Serial.println("START");
	Serial.println();

	adxl.powerOn();
	adxl.setRangeSetting(4);       //Definir el rango, valores 2, 4, 8 o 16
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

/////////
// SPI //
/////////

/* #include <SPI.h> */

/* //Assign the Chip Select signal to pin 10. */
/* int CS=10; */

/* //This is a list of some of the registers available on the ADXL345. */
/* //To learn more about these and the rest of the registers on the ADXL345, read the datasheet! */
/* char POWER_CTL = 0x2D; */
/* char DATA_FORMAT = 0x31; */
/* char DATAX0 = 0x32; */
/* char DATAX1 = 0x33; */
/* char DATAY0 = 0x34; */
/* char DATAY1 = 0x35; */
/* char DATAZ0 = 0x36; */
/* char DATAZ1 = 0x37; */

/* //This buffer will hold values read from the ADXL345 registers. */
/* char values[10]; */
/* //These variables will be used to hold the x,y and z axis accelerometer values. */
/* int x,y,z; */

/* void setup(){ */

  /* SPI.begin(); */

  /* SPI.setDataMode(SPI_MODE3); */

  /* Serial.begin(115200); */


  /* pinMode(CS, OUTPUT); */

  /* digitalWrite(CS, HIGH); */


  /* writeRegister(DATA_FORMAT, 0x01); */

  /* writeRegister(POWER_CTL, 0x08);  */

  /* Serial.println("START"); */
/* } */

/* void loop(){ */


  /* readRegister(DATAX0, 6, values); */

/* //The ADXL345 gives 10-bit acceleration values, but they are stored as bytes (8-bits). To get the full value, two bytes must be combined for each axis. */

  /* x = ((int)values[1]<<8)|(int)values[0]; */

  /* y = ((int)values[3]<<8)|(int)values[2]; */

  /* z = ((int)values[5]<<8)|(int)values[4]; */


  /* Serial.print(millis()); */
  /* Serial.print(","); */
  /* Serial.print(x, DEC); */
  /* Serial.print(","); */
  /* Serial.print(y, DEC); */
  /* Serial.print(","); */
  /* Serial.println(z, DEC); */
/* } */

/* //This function will write a value to a register on the ADXL345. */
/* //Parameters: */
/* //  char registerAddress - The register to write a value to */
/* //  char value - The value to be written to the specified register. */
/* void writeRegister(char registerAddress, char value){ */

  /* digitalWrite(CS, LOW); */

  /* SPI.transfer(registerAddress); */

  /* SPI.transfer(value); */

  /* digitalWrite(CS, HIGH); */
/* } */

/* //This function will read a certain number of registers starting from a specified address and store their values in a buffer. */
/* //Parameters: */
/* //  char registerAddress - The register addresse to start the read sequence from. */
/* //  int numBytes - The number of registers that should be read. */
/* //  char * values - A pointer to a buffer where the results of the operation should be stored. */
/* void readRegister(char registerAddress, int numBytes, char * values){ */

  /* char address = 0x80 | registerAddress; */

  /* if(numBytes > 1)address = address | 0x40; */


  /* digitalWrite(CS, LOW); */

  /* SPI.transfer(address); */

  /* for(int i=0; i<numBytes; i++){ */
    /* values[i] = SPI.transfer(0x00); */
  /* } */

  /* digitalWrite(CS, HIGH); */
/* } */
