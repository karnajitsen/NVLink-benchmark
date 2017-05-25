#include<stdio.h>
#include<math.h>
#include <time.h>
#include <string.h>
#include<stdlib.h>
#include<iostream>
#include "parameters.h"
using namespace std;
int main(int argc, char * argv[])
{

   FILE *fp;
   long int chelem,start, end, startStr, endStr , i,stride;//, n, p, pr = 4, csize = 8;
   //stride = atoi(argv[1]);
   char filename[100];
   strcpy(filename, "execute");
   strcat(filename,".sh");
     fp = fopen(filename, "w+");
   
   if(tpMode == 'Y')
   {	   
   if(startUnit == 'B')
     { 
	   if (startSize%8 == 0)
	  	 start = startSize/8;
	   else 
		 start = startSize/8 + 1;
      }
   else if(startUnit == 'K')
   {
   	if(startSize%8 == 0)
		start = (startSize*1024)/8;
	else
		start =(startSize*1024)/8 + 1;
   }
   else if(startUnit == 'M')
   {
	 if(startSize%8 == 0)
		 start = ((long int)startSize*1024*1024)/8;
	 else
		 start = ((long int)startSize * 1024 * 1024)/ 8 + 1;
   }
   else if(startUnit == 'G')
    {
	    if(startSize%8 == 0)
		    start = ((long int) startSize * 1024 * 1024 * 1024)/8;
	    else
		    start = ((long int) startSize * 1024 * 1024 * 1024)/ 8 + 1;
   }


    if(endUnit == 'B')
    {
      if (endSize%8 == 0)
            end = endSize/8;
       else
            end = endSize/8 + 1;
     }
    else if(endUnit == 'K')
	   {
	    if(endSize%8 == 0)
	      end = (endSize*1024)/8;
            else
 	      end =(endSize*1024)/8 + 1;
	   }
    else if(endUnit == 'M')
    {
     if(endSize%8 == 0)
        end = ((long int) endSize*1024*1024)/8;
     else
        end = ((long int) endSize * 1024 * 1024)/ 8 + 1;
     }
    else if(endUnit == 'G')
    {
      if(endSize%8 == 0)
	   	end = ((long int) endSize * 1024 * 1024 * 1024)/8;
     else
	end =((long int) endSize * 1024 * 1024 * 1024)/ 8 + 1;
   }
   }
   int lengthdtype ;
     if(tpMode == 'Y')
        lengthdtype = sizeof(dType);
     else
        lengthdtype = sizeof(dLatType);
  
  	 if(startStrideUnit == 'B')
     {
           if (startStride%lengthdtype == 0)
                 startStr = startStride/lengthdtype;
           else
                 startStr = startStride/lengthdtype + 1;
      }
   else if(startStrideUnit == 'K')
   {
        if(startStride%lengthdtype == 0)
                startStr = (startStride*1024)/lengthdtype;
        else
                startStr =(startStride*1024)/lengthdtype + 1;
   }
   else if(startStrideUnit == 'M')
   {
         if(startStride%lengthdtype == 0)
                 startStr = ((long int)startStride*1024*1024)/lengthdtype;
         else
                 startStr = ((long int)startStride * 1024 * 1024)/ lengthdtype + 1;
   }
   else if(startStrideUnit == 'G')
    {
            if(startStride%lengthdtype == 0)
                    startStr = ((long int) startStride * 1024 * 1024 * 1024)/lengthdtype;
            else
                    startStr = ((long int) startStride * 1024 * 1024 * 1024)/ lengthdtype + 1;
   }


    if(endStrideUnit == 'B')
    {
      if (endStride%lengthdtype == 0)
            endStr = endStride/lengthdtype;
       else
            endStr = endStride/lengthdtype + 1;
     }
    else if(endStrideUnit == 'K')
           {
            if(endStride%lengthdtype == 0)
              endStr = (endStride*1024)/lengthdtype;
   		else
              endStr =(endStride*1024)/lengthdtype + 1;
           }
    else if(endStrideUnit == 'M')
    {
     if(endStride%lengthdtype == 0)
        endStr = ((long int) endStride*1024*1024)/lengthdtype;
     else
        endStr = ((long int) endStride * 1024 * 1024)/ lengthdtype + 1;
     }
    else if(endStrideUnit == 'G')
    {
      if(endStride%lengthdtype == 0)
                endStr = ((long int) endStride * 1024 * 1024 * 1024)/lengthdtype;
     else
        endStr =((long int) endStride * 1024 * 1024 * 1024)/ lengthdtype + 1;
   }
 
    
  if(startStr == 0)
          startStr=1;
 
   
   if(start == 0)
	  start=32; 
    i = 1;
   // start = 32;
   if(tpMode == 'Y')
   {
   stride = startStr;
     fprintf(fp, "cd data\n");
    fprintf(fp, "mv result-stream-bw-um.txt result-stream-bw-um.bk_%d_%ld\n",(int)time(NULL),stride);
   fprintf(fp, "mv result-stream-bw-um.csv result-stream-bw-um.csv_bk_%d_%ld\n",(int)time(NULL),stride);
   fprintf(fp, "mv result-stream-bw-nonum.txt result-stream-bw-nonum.bk_%d_%ld\n",(int)time(NULL),stride);
   fprintf(fp, "mv result-stream-bw-nonum.csv result-stream-bw.nonum.csv_bk_%d_%ld\n",(int)time(NULL),stride);
   fprintf(fp, "mv result-stream-bw-memcpy.txt result-stream-bw-memcpy.bk_%d_%ld\n",(int)time(NULL),stride);
   fprintf(fp, "mv result-stream-bw-memcpy.csv result-stream-bw-memcpy.csv_bk_%d_%ld\n",(int)time(NULL),stride);
    fprintf(fp, "cd ..\n");
  long long int size;
  long long thread = startomp;
  while(stride <= 2048)
  {
    if(start == 0)
          size=32;
    else size = start;
    while ( size <= end)
    { 
      while(startomp <= endomp)
      {   
      fprintf(fp, "numactl -C 1-%d ./nvprog %ld %ld %ld 1\n", 1+startomp ,size, stride,startomp);
      startomp +=inromp;
      }
//      fprintf(fp, "./nvprog %ld %ld 80 1\n", size/stride, stride);
      startomp = thread;
      size = size*inc_size;
    }
    stride = stride * inc_stride;
   }
  } 

 
   if(latMode == 'Y')
   {
    chelem = startChelem;
    fprintf(fp, "cd data\n");
   fprintf(fp, "mv result-pchase-lat-um.txt result-pchase-lat-um.bk_%d_%ld\n",(int)time(NULL),chelem);
   fprintf(fp, "mv result-pchase-lat-um.csv result-pchase-lat-um.csv_bk_%d_%ld\n",(int)time(NULL),chelem);
   fprintf(fp, "mv result-pchase-lat-nonum.txt result-pchase-lat-nonum.bk_%d_%ld\n",(int)time(NULL),chelem);
   fprintf(fp, "mv result-pchase-lat-nonum.csv result-pchase-lat-nonum.csv_bk_%d_%ld\n",(int)time(NULL),chelem);
     fprintf(fp, "cd ..\n");
   while ( chelem <= endChelem)
    {
     stride = startStr;
     while(stride <= endStr)
    {
      fprintf(fp, "numactl -C 40 ./nvprog %ld %ld 1 0 1\n", chelem, stride);
      stride = stride*inc_stride;
    }
      if(chaseIncSign == '*')
      chelem  = chelem * inc_chelem;
      if(chaseIncSign == '+')
       chelem = chelem + inc_chelem;
     if(chaseIncSign == '-')
       chelem = chelem - inc_chelem;

   }
  }

   //fprintf(fp, "gnuplot plot.in\n");
   fclose(fp);
   return 0;
}
