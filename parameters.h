#include<iostream>
#include <string>
#include <vector>
#define NOOFSTREAM 8
#define MAXKERMEMMAPS 6
using namespace std;

typedef double dType;
typedef int dLatType;
const std::string accPattern = "LINEAR";
const std::string memMode = "NONUM";
const char latMode =  'Y';
const char memAdvise =  'N';
const char tpMode = 'N';
const char cudaMemMode = 'N';
const char exp_thread = 'N';
const int startStride = 4;
const char startStrideUnit = 'B';
const int endStride = 512;
const char endStrideUnit = 'K';
const int startChelem = 128;
const int endChelem = 16384;
const char chaseIncSign = '*';
const int startSize = 32;
const char startUnit = 'M';
const int endSize = 8;
const char endUnit = 'G';
int startomp = 1;
int endomp = 1;
const int inromp = 4;
const int inc_size = 2;
const int inc_stride = 2;
const int inc_chelem = 2;
const std::string kloc[6] = {"C0","","","","",""};
unsigned int num_times = 1;
const std::string mloc[6][6] = {{"C0","","","","",""},{"","","","","",""},{"","","","","",""},{"","","","","",""},{"","","","","",""},{"","","","","",""}};
std::string  strMode[8] = {"READ","WRITE","COPY","ADD","TRIAD","QUADAD","PENTAD","HEXAD"};
const int startBlock = 1024;
const int endBlock = 1024;
const int iterations = 1;
std::vector<string> streamMode;
