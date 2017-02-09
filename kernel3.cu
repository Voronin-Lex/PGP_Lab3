#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define BLOCKSIZE 512


__host__ unsigned int getMax(unsigned int*  Data, int n)
{
   unsigned  int mx = Data[0];
    for (int i = 1; i < n; i++)
        if (Data[i] > mx)
            mx = Data[i];
    return mx;
}

__global__ void FixUpScan(unsigned int* PreScan, unsigned int* PreScannedAuxiliary, unsigned int Size)
{
	int tid = threadIdx.x;
	int BlockOffset=0;

	for(int offset = 2*blockIdx.x*blockDim.x; offset<Size; offset+=2*blockDim.x*gridDim.x)
 {
	 if((BlockOffset*gridDim.x+blockIdx.x)<(Size+2*BLOCKSIZE-1)/(2*BLOCKSIZE))
	{
		if ((offset+2*tid) < Size)
		PreScan[offset+2*tid]+=PreScannedAuxiliary[BlockOffset*gridDim.x+blockIdx.x];

		if ((offset+2*tid+1) < Size)
		PreScan[offset+2*tid+1]+=PreScannedAuxiliary[BlockOffset*gridDim.x+blockIdx.x];
	}

	BlockOffset++;
  }

}

__global__ void GetDiff_N_Sn(unsigned char* Bits, unsigned int* PreScan, unsigned int* Diff_N_Sn, int Size)
{
	*Diff_N_Sn = Size - PreScan[Size-1];
	if(Bits[Size-1]==1) (*Diff_N_Sn)--;
	Bits[Size-1]=Bits[Size-1];   //для компилятора


}
 


__global__ void Sort(unsigned int* InData, unsigned int* PreScan, unsigned int* OutData, unsigned char* Bits, unsigned int* Diff_N_Sn, int Size)
{
	
  
	for(int index = blockIdx.x*blockDim.x+threadIdx.x; index<Size; index+=blockDim.x*gridDim.x)
   {
	if(index<Size)
	{

	if(Bits[index]==0)
	{
		OutData[index - PreScan[index]] = InData[index];
	}

	else
	  OutData[PreScan[index]+(*Diff_N_Sn)] = InData[index];
      		 
     }

   }
}

__global__ void Exchange(unsigned int* InData, unsigned int* OutData, int Size)
{
	for(int index=blockIdx.x*blockDim.x+threadIdx.x; index<Size; index+=blockDim.x*gridDim.x)
	{
		InData[index]=OutData[index];	
	}
}

__global__ void KernelPrescan(unsigned int* Data, unsigned char* Bits, unsigned int* PreScan, unsigned int* Auxiliary, int Size, int bit)   
{
	extern __shared__	unsigned int Tmp[];

	int tid = threadIdx.x;
	int AuxiliaryIndex=0;

	for(int OffsetTid = 2*blockIdx.x*blockDim.x; OffsetTid<Size; OffsetTid+=2*blockDim.x*gridDim.x)
  {
	int offset=1;

	if(OffsetTid+tid<Size)
	 {
		 Tmp[tid] =  (Data[OffsetTid+tid]>>bit)&1;
	     Bits[OffsetTid+tid]=(Data[OffsetTid+tid]>>bit)&1;
	 }
	else
     Tmp[tid] = 0;



	if(OffsetTid+tid+blockDim.x<Size)
	{
		Tmp[tid+blockDim.x] =  (Data[OffsetTid+tid+blockDim.x]>>bit)&1;
		Bits[OffsetTid+tid+blockDim.x] = (Data[OffsetTid+tid+blockDim.x]>>bit)&1;
	}
	 else
     Tmp[tid+blockDim.x] = 0;
	


	for (int d = blockDim.x; d > 0; d>>=1)
	{
		__syncthreads();

		if(tid<d)
		{
	        int ai = offset*(2*tid+1)-1;	
		    int bi = offset*(2*tid+2)-1;	

			Tmp[bi]+=Tmp[ai];
		}
		offset*=2;
	}

	if(tid==0)  
    
	{  if((gridDim.x*AuxiliaryIndex+blockIdx.x)<(Size+2*BLOCKSIZE-1)/(2*BLOCKSIZE))
	    Auxiliary[gridDim.x*AuxiliaryIndex+blockIdx.x]=Tmp[2*blockDim.x-1];
	   
	    Tmp[2*blockDim.x-1]=0; 
	} 

	for(int d=1; d<2*blockDim.x; d*=2)
	{
		offset>>=1;
		__syncthreads();


		if(tid<d)
		{
		   int ai = offset*(2*tid+1)-1;	
		   int bi = offset*(2*tid+2)-1;	
		
		   int t = Tmp[ai];
		   Tmp[ai]=Tmp[bi];
		   Tmp[bi]+=t;
		
		}	
	}

	__syncthreads();


	if((OffsetTid+2*tid)<Size)                         
	PreScan[OffsetTid+2*tid] =  Tmp[2*tid];

	if((OffsetTid+2*tid+1)<Size)
	PreScan[OffsetTid+2*tid+1] = Tmp[2*tid+1]; 

	__syncthreads();
 
	AuxiliaryIndex++;
	}
}


__global__ void KernelPrescanRecursive(unsigned int* PreScan, unsigned int* Auxiliary, int Size, int LastLevel)  
{
	extern __shared__	unsigned int Tmp[];

	int tid = threadIdx.x;
	int AuxiliaryIndex=0;

	for(int OffsetTid = 2*blockIdx.x*blockDim.x; OffsetTid<Size; OffsetTid+=2*blockDim.x*gridDim.x)
  {
	int offset=1;

	if(OffsetTid+tid<Size)
	 {
		 Tmp[tid] =  PreScan[OffsetTid+tid];
	 }
	else
     Tmp[tid] = 0;



	if(OffsetTid+tid+blockDim.x<Size)
	{
		Tmp[tid+blockDim.x] =  PreScan[OffsetTid+tid+blockDim.x];
	}
	 else
     Tmp[tid+blockDim.x] = 0;
	


	for (int d = blockDim.x; d > 0; d>>=1)
	{
		__syncthreads();

		if(tid<d)
		{
	        int ai = offset*(2*tid+1)-1;	
		    int bi = offset*(2*tid+2)-1;	

			Tmp[bi]+=Tmp[ai];
		}
		offset*=2;
	}

	if(tid==0)  
    
	{  if( ((gridDim.x*AuxiliaryIndex+blockIdx.x)<(Size+2*BLOCKSIZE-1)/(2*BLOCKSIZE)) && LastLevel==0)
	    Auxiliary[gridDim.x*AuxiliaryIndex+blockIdx.x]=Tmp[2*blockDim.x-1];
	   
	    Tmp[2*blockDim.x-1]=0; 
	} 

	for(int d=1; d<2*blockDim.x; d*=2)
	{
		offset>>=1;
		__syncthreads();


		if(tid<d)
		{
		   int ai = offset*(2*tid+1)-1;	
		   int bi = offset*(2*tid+2)-1;	
		
		   int t = Tmp[ai];
		   Tmp[ai]=Tmp[bi];
		   Tmp[bi]+=t;
		
		}	
	}

	__syncthreads();


	if((OffsetTid+2*tid)<Size)                         
	PreScan[OffsetTid+2*tid] =  Tmp[2*tid];

	if((OffsetTid+2*tid+1)<Size)
	PreScan[OffsetTid+2*tid+1] = Tmp[2*tid+1]; 

	__syncthreads();
 
	AuxiliaryIndex++;
	}
}


__host__ void PreScanRecursive(unsigned int** ListAux, unsigned int* ListAuxSize, int CountAuxiliary, int Depth)
{
	if(CountAuxiliary==0) return;
	int LastLevel=0;

	if(ListAuxSize[Depth]<=2*BLOCKSIZE)     
	{
		LastLevel=1;
		KernelPrescanRecursive<<<512, BLOCKSIZE, 2*BLOCKSIZE*sizeof(unsigned int)>>>(ListAux[Depth], NULL, ListAuxSize[Depth], LastLevel);	
	}

	else
	{
		KernelPrescanRecursive<<<512, BLOCKSIZE, 2*BLOCKSIZE*sizeof(unsigned int)>>>(ListAux[Depth], ListAux[Depth+1], ListAuxSize[Depth], LastLevel);	
	}

	PreScanRecursive(ListAux, ListAuxSize, CountAuxiliary-1, Depth+1);

	if(LastLevel==0)
		FixUpScan<<<512,BLOCKSIZE>>>(ListAux[Depth], ListAux[Depth+1], ListAuxSize[Depth]);
	
}

int main()
{
	int Size;
	
	fread(&Size, sizeof(int), 1, stdin);
	
	unsigned int*  Data = (unsigned int*)malloc(Size*sizeof(unsigned int));
	fread(Data, Size*sizeof(unsigned int), 1, stdin);
     
	unsigned int m = getMax(Data, Size);   

	unsigned int*  PreScan;        
	unsigned char*  Bits;          
	unsigned int* Dev_Data;        
	unsigned int* Diff_N_Sn;       
	unsigned int* OutData;
	


	cudaMalloc((void**)&Dev_Data, Size*sizeof(unsigned int));
	cudaMalloc((void**)&PreScan, Size*sizeof(unsigned int));
	cudaMalloc((void**)&Bits, Size*sizeof(unsigned char));
	cudaMalloc((void**)&Diff_N_Sn, sizeof(unsigned int));
	cudaMalloc((void**)&OutData, Size*sizeof(unsigned int));
	cudaMemcpy(Dev_Data, Data, Size*sizeof(unsigned int), cudaMemcpyHostToDevice);


	int CountAuxiliary = 0;    
	int PrevAuxSize = Size;
	int NextAuxSize=0;

    do 
	{
		NextAuxSize = (PrevAuxSize+2*BLOCKSIZE-1)/(2*BLOCKSIZE);  
		PrevAuxSize = NextAuxSize;
		CountAuxiliary++;
	} while(NextAuxSize >= 2*BLOCKSIZE);


	unsigned int* ListAuxSize = (unsigned int*)malloc(CountAuxiliary*sizeof(unsigned int));
	
	unsigned int** ListAux;

	ListAux = (unsigned int**)malloc(CountAuxiliary*sizeof(unsigned int*));

	 PrevAuxSize = Size;
	 NextAuxSize=0;

	for(int i=0; i<CountAuxiliary; i++)
	{
	  NextAuxSize = (PrevAuxSize+2*BLOCKSIZE-1)/(2*BLOCKSIZE);
	  ListAuxSize[i] = NextAuxSize;
	  cudaMalloc((void**)&ListAux[i], NextAuxSize*sizeof(unsigned int));
	  //cudaMemcpy(ListAux[i], ListSupport[i], NextAuxSize*sizeof(unsigned int), cudaMemcpyHostToDevice);
	  PrevAuxSize = NextAuxSize;
	}

	for (unsigned int bit = 0; (m>>bit) > 0; bit++)
	{
		KernelPrescan<<<512,BLOCKSIZE, 2*BLOCKSIZE*sizeof(unsigned int)>>>(Dev_Data, Bits, PreScan, ListAux[0], Size, bit);   //invoke CountSort by every bit
		PreScanRecursive(ListAux, ListAuxSize, CountAuxiliary, 0);
		FixUpScan<<<512,BLOCKSIZE>>>(PreScan, ListAux[0], Size);
		GetDiff_N_Sn<<<1,1>>>(Bits, PreScan, Diff_N_Sn, Size);
		Sort<<<512,BLOCKSIZE>>>(Dev_Data, PreScan, OutData, Bits, Diff_N_Sn, Size);
		Exchange<<<512,BLOCKSIZE>>>(Dev_Data, OutData, Size);

		if ((m>>bit)==1)  
		{
			break;
		}
	}


	cudaMemcpy(Data, Dev_Data, Size*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	 
	fwrite(Data, Size*sizeof(unsigned int), 1, stdout);

    return 0;
}
