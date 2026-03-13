

// // importing libraries
// #include <ESP8266WiFi.h>
// #include <ESP8266WebServer.h>
// #include <Wire.h>
// #include <LiquidCrystal_I2C.h> 


// // lcd address
// LiquidCrystal_I2C lcd(0x27, 16, 2);

// // wifi setup
// const char* ssid = "SHUKLA4G";
// const char* password = "Vjordan2002";

// ESP8266WebServer server(80);

// // Define pins for the ultrasonic sensor
// const int trigPin = D1;
// const int echoPin = D2;

// // Global variable to store the latest distance
// int distance = 0;

// // Update the distance measurement
// void updateDistance() {
//   long duration;
  
//   // Trigger the ultrasonic sensor
//   digitalWrite(trigPin, LOW);
//   delayMicroseconds(2);
//   digitalWrite(trigPin, HIGH);
//   delayMicroseconds(10);
//   digitalWrite(trigPin, LOW);

//   // Measure the pulse
//   duration = pulseIn(echoPin, HIGH);

//   // Calculate distance
//   distance = (duration / 2.0) * 0.0344;// Convert to centimeters
//   if(distance>810){
//     distance = 0;
//   }
//   else{
//     return;
//   } 
//   Serial.println(distance);
// }

// // Handle HTTP GET request to /distance
// void handleDistance() {
//   String message = "Distance: " + String(distance) + " cm";
//   server.send(200, "text/plain", message);
// }

// void setup() {
//   // band rate
//   Serial.begin(115200);
//   WiFi.begin(ssid, password);

//   while (WiFi.status() != WL_CONNECTED) {
//     delay(500);
//     Serial.print(".");
//   }

//   Serial.println("Connected to WiFi");
//   Serial.print("http://");
//   Serial.println(WiFi.localIP());

//   // Initialize ultrasonic sensor pins
//   pinMode(trigPin, OUTPUT);
//   pinMode(echoPin, INPUT);

//   // Define route and handler
//   server.on("/distance", HTTP_GET, handleDistance);

//   server.begin();

//   Wire.begin(2,0);

//   // display of distance
//   lcd.init();   // initializing the LCD
//   lcd.backlight(); // Enable or Turn On the backlight 
//   lcd.clear();
//   lcd.setCursor(0,0);
//   lcd.print("Vicinity sensor"); // Start Printing
//   delay(1500);
//   lcd.clear();
//   lcd.setCursor(0,0);
//   lcd.print("Distance: ");

// }

// void loop() {
//   server.handleClient();//initializing the http server
//   updateDistance();// Update the distance measurement every loop iteration
//   lcd.setCursor(0,1);
//   lcd.print(distance);
//   lcd.print(" cm ");
 
//  // Update the distance measurement every loop iteration
//   delay(2000);       // Wait for 2 seconds before updating again
// }


// importing libraries
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h> 


// lcd address
LiquidCrystal_I2C lcd(0x27, 16, 2);

// wifi setup
const char* ssid = "Galaxy M31sB1C1";
const char* password = "xfwh9956";

ESP8266WebServer server(80);

// -------- IR SENSOR --------
const int irPin = A0;

// -------- Ultrasonic (COMMENTED) --------
// const int trigPin = D1;
// const int echoPin = D2;


// Global variable to store the latest distance
int distance = 0;


// Update the distance measurement
void updateDistance() {

  // -------- IR SENSOR READING --------
  int sensorValue = analogRead(irPin);

  float voltage = sensorValue * (3.3 / 1023.0);

  // approximate formula for GP2Y0A02YK0F
  float dist = 27.86 / (voltage - 0.42);

  if (dist < 0 || dist > 150) {
    distance = 0;
  } else {
    distance = (int)dist;
  }

  Serial.print("Distance: ");
  Serial.println(distance);

  // -------- ULTRASONIC CODE (COMMENTED) --------
  /*
  long duration;
  
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH);

  distance = (duration / 2.0) * 0.0344;

  if(distance>810){
    distance = 0;
  }
  else{
    return;
  }
  Serial.println(distance);
  */
}


// Handle HTTP GET request to /distance
void handleDistance() {
  String message = "Distance: " + String(distance) + " cm";
  server.send(200, "text/plain", message);
}


void setup() {

  Serial.begin(115200);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("Connected to WiFi");
  Serial.print("http://");
  Serial.println(WiFi.localIP());


  // -------- Ultrasonic pins (COMMENTED) --------
  /*
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  */


  // Define route and handler
  server.on("/distance", HTTP_GET, handleDistance);

  server.begin();

  Wire.begin(2,0);


  // LCD display
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Vicinity sensor");
  delay(1500);

  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Distance: ");
}


void loop() {

  server.handleClient();

  updateDistance();

  lcd.setCursor(0,1);
  lcd.print(distance);
  lcd.print(" cm ");

  delay(2000);
}