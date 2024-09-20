//程序基本信息：
/*
能够实现GSR与PPG传感器同时采集信号
采样率：
GSR:200Hz
PPG:50Hz
波特率：
9600
支持修改端口号
*/

//连接方式：
/*
GSR:
"GND"->GND
"VCC"->5V
"OUT"->A0

PPG:
"-"->GND
"+"->3.3V
"S"->A1
*/

/*
串口行数据说明：
G皮肤电原始数据

S   脉搏波原始数据
Q   脉搏波原始数据计算得到的心跳间隔数据
B   心跳间隔数据计算得到的心率数据
*/
// 引脚定义
const int GSR_LED = 13;     // GSR的闪烁LED引脚
const int GSR_PIN = A0;     // GSR传感器接到A0
const int PULSE_PIN = A1;   // PPG传感器接到A1
const int BLINK_PIN = 12;   // PPG的闪烁LED引脚
const int FADE_PIN = 5;     // PPG的渐变LED引脚

// GSR 传感器变量
/*
int gsrThreshold = 0;       //皮肤电平均值
*/
int gsrSensorValue;         //皮肤电瞬时值

// 脉搏波传感器变量
volatile int BPM;           //Beat Per Min 心率
volatile int Signal;        //Signal 脉搏信号
volatile int IBI = 600;     //心跳间隔时间（单位：ms）
volatile boolean Pulse = false;
volatile boolean QS = false;
int fadeRate = 0;

// 设置
void setup() {
  // 初始化串口
  Serial.begin(9600);
  
  // 初始化GSR传感器
  pinMode(GSR_LED, OUTPUT);       //GSR的LED来判断是否写入
  digitalWrite(GSR_LED, LOW);

  /*
  long sum = 0;
  for (int i = 0; i < 500; i++) {
    gsrSensorValue = analogRead(GSR_PIN);//读取GSR引脚获取模拟输入
    sum += gsrSensorValue;
    delay(5);                      //每次循环延时5ms，实现200Hz采样
  }
  */

  /*
  gsrThreshold = sum / 500;        //每2.5s得到一次基准值（临界值）？
  Serial.print("GSR threshold = ");
  Serial.println(gsrThreshold);
  */
  
  // 初始化脉搏波传感器
  pinMode(BLINK_PIN, OUTPUT);
  pinMode(FADE_PIN, OUTPUT);
  interruptSetup();
}

// 主循环
void loop() {
  // GSR 数据采集和处理
  gsrSensorValue = analogRead(GSR_PIN);
  Serial.print("G");         //串口行输出GSR的值
  Serial.println(gsrSensorValue);
  //下面利用一段时间内皮肤电变化来判断是否有情绪变化
  //int temp = gsrThreshold - gsrSensorValue;
  //如果当前皮肤电数值与最近2.5s内基准值差值大于60则视为情绪变化，在串口行输出


  /*
  if (abs(temp) > 60) {
    gsrSensorValue = analogRead(GSR_PIN);
    temp = gsrThreshold - gsrSensorValue;
    if (abs(temp) > 60) {
      digitalWrite(GSR_LED, HIGH);
      Serial.println("Emotion Changes Detected!");
      delay(3000);
      digitalWrite(GSR_LED, LOW);
      delay(1000);
    }
  }
  */
  
  // 脉搏波传感器数据采集和处理
  sendDataToProcessing('S', Signal);  //S代表原始脉搏数据
  if (QS == true) {
    fadeRate = 255;
    sendDataToProcessing('B', BPM);   //B代表当前心率
    sendDataToProcessing('Q', IBI);   //Q代表心跳间隔时间
    QS = false;
  }
  ledFadeToBeat();
  delay(20); // 延时20ms，采样率50Hz
}

// 脉搏波传感器LED渐变
// 范围：0~255
void ledFadeToBeat() {
  fadeRate -= 15;
  fadeRate = constrain(fadeRate, 0, 255);
  analogWrite(FADE_PIN, fadeRate);
}

// 函数
// 通过串口向处理器（PC）发送数据
// 数据格式：（S/B/Q）＋ 数值
void sendDataToProcessing(char symbol, int data) {
  Serial.print(symbol);
  Serial.println(data);
}

// 中断设置
void interruptSetup() {
  TCCR2A = 0x02;
  TCCR2B = 0x06;
  OCR2A = 0x7C;
  TIMSK2 = 0x02;
  sei();
}

// 中断服务程序
// 原理：采集到连续的脉搏信号后进行分析：峰峰值之间为心跳间隔，据此计算心率
ISR(TIMER2_COMPA_vect) {
  cli();
  Signal = analogRead(PULSE_PIN);
  static unsigned long sampleCounter = 0;
  static unsigned long lastBeatTime = 0;
  static int P = 512;
  static int T = 512;
  static int thresh = 512;
  static int amp = 100;
  static boolean firstBeat = true;
  static boolean secondBeat = false;
  static int rate[10];
  sampleCounter += 2;
  int N = sampleCounter - lastBeatTime;
  
  if (Signal < thresh && N > (IBI / 5) * 3) {
    if (Signal < T) {
      T = Signal;
    }
  }

  if (Signal > thresh && Signal > P) {
    P = Signal;
  }

  if (N > 250) {
    if ((Signal > thresh) && (Pulse == false) && (N > (IBI / 5) * 3)) {
      Pulse = true;
      digitalWrite(BLINK_PIN, HIGH);
      IBI = sampleCounter - lastBeatTime;
      lastBeatTime = sampleCounter;
      
      if (secondBeat) {
        secondBeat = false;
        for (int i = 0; i <= 9; i++) {
          rate[i] = IBI;
        }
      }

      if (firstBeat) {
        firstBeat = false;
        secondBeat = true;
        sei();
        return;
      }

      long runningTotal = 0;
      for (int i = 0; i <= 8; i++) {
        rate[i] = rate[i + 1];
        runningTotal += rate[i];
      }

      rate[9] = IBI;
      runningTotal += rate[9];
      runningTotal /= 10;
      BPM = 60000 / runningTotal;
      QS = true;
    }
  }

  if (Signal < thresh && Pulse == true) {
    digitalWrite(BLINK_PIN, LOW);
    Pulse = false;
    amp = P - T;
    thresh = amp / 2 + T;
    P = thresh;
    T = thresh;
  }

  if (N > 2500) {
    thresh = 512;
    P = 512;
    T = 512;
    lastBeatTime = sampleCounter;
    firstBeat = true;
    secondBeat = false;
  }

  sei();
}

