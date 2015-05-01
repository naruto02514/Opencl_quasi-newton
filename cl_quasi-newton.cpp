#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

#define M 2
#define min 1.0e-5
#define MAX_SOURCE_SIZE (0x100000)

double tstart, tstop, ttime; 
int i,j,k,loop,flag;
int m =(int)M;

float alpha;
float d[M];
float *H=(float *)malloc(M*M*sizeof(float));
float *H1=(float *)malloc(M*M*sizeof(float));
float *b=(float *)malloc(M*sizeof(float));
float *b1=(float *)malloc(M*sizeof(float));
float *x1=(float *)malloc(M*sizeof(float));
float *x2=(float *)malloc(M*sizeof(float));

float *HysT=(float *)malloc(M*M*sizeof(float));
float *sHyT=(float *)malloc(M*M*sizeof(float));
float *ssT=(float *)malloc(M*M*sizeof(float));
float *y=(float *)malloc(M*sizeof(float));
float *s=(float *)malloc(M*sizeof(float));
float *Hy=(float *)malloc(M*sizeof(float));
float *sTy=(float *)malloc(sizeof(float));
float *yTHy=(float *)malloc(sizeof(float));

cl_platform_id platform_id=NULL; 
cl_device_id device_id=NULL;
cl_context context=NULL;
cl_command_queue command_queue=NULL;
cl_mem Hmobj=NULL;
cl_mem H1mobj=NULL;
cl_mem bmobj=NULL;
cl_mem b1mobj=NULL;
cl_mem x1mobj=NULL;
cl_mem x2mobj=NULL;

cl_mem HysTmobj=NULL;
cl_mem sHyTmobj=NULL;
cl_mem ssTmobj=NULL;
cl_mem ymobj=NULL; 
cl_mem smobj=NULL;
cl_mem Hymobj=NULL;
cl_mem sTymobj=NULL;
cl_mem yTHymobj=NULL; 

cl_program program=NULL;
cl_kernel kernel[4]={NULL,NULL,NULL,NULL};
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
cl_int ret;
size_t source_size;
char *source_str;

float fx1(float x,float y){
	return 2*(1-200*y)*x+400*x*x*x-2;
}

float fx2(float x,float y){
	return 200*(y-x*x);
}

float f(float x,float y){
	return 100*((y-x*x)*(y-x*x))+((1-x)*(1-x));
}

void initial(){
	loop=0;
	flag=0;
	x1[0]=-2;
	x1[1]=2;
	H[0*M+0]=1;
	H[0*M+1]=0;
	H[1*M+0]=0;
	H[1*M+1]=1;
	b[0]=fx1(x1[0],x1[1]);
	b[1]=fx2(x1[0],x1[1]);
}

void hk(){
	for(i=0;i<M;i++){
		d[i]=0;
		for(j=0;j<M;j++){
			d[i]+=-(H[i*M+j])*b[j];
		}
	}
}

void alpha_update(){
	float FTD,ff;
	FTD=0;
	alpha=1;
	ff=f(x1[0],x1[1]); //f(x)
	for(i=0;i<M;i++){
		FTD+=b[i]*d[i]; //Þf(x)*dk
	}
	while(f(x1[0]+alpha*d[0],x1[1]+alpha*d[1]) >= ff+0.5*alpha*FTD){
		if(alpha<1.0e-6){
			alpha=1.0e-6;
			break;
		}
		else
			alpha*=0.5;
	}
}

void x2_update(){
	for(int i=0;i<M;i++){
		x2[i]=x1[i]+alpha*d[i]; //x(k+1)=xk+ƒ¿*dk
	}
}

void flagx(){
	if(fabs(b[0])<=min && fabs(b[1])<=min){
		flag=1;
	}
}

void cl_start(){
	FILE *fp;
	const char fileName[]="./cl_quasi-newton.cl";
	fopen_s(&fp,fileName,"r");
	if(!fp){
		fprintf(stderr,"Failed to load kernel.\n"); 
		exit(1);
	}
	source_str = (char *)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str,1,MAX_SOURCE_SIZE,fp);
	fclose(fp);
	
	//Get GPU ID
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

	//Create context
	context=clCreateContext(NULL,1,&device_id,NULL,NULL,&ret);

	//Create command queue
	command_queue = clCreateCommandQueue(context,device_id,0,&ret);

	//Create buffer object
	Hmobj=clCreateBuffer(context,CL_MEM_READ_WRITE,M*M*sizeof(float),NULL,&ret);
	H1mobj=clCreateBuffer(context,CL_MEM_READ_WRITE,M*M*sizeof(float),NULL,&ret);
	bmobj=clCreateBuffer(context,CL_MEM_READ_WRITE,M*sizeof(float),NULL,&ret);
	b1mobj=clCreateBuffer(context,CL_MEM_READ_WRITE,M*sizeof(float),NULL,&ret);
	x1mobj=clCreateBuffer(context,CL_MEM_READ_WRITE,M*sizeof(float),NULL,&ret);
	x2mobj=clCreateBuffer(context,CL_MEM_READ_WRITE,M*sizeof(float),NULL,&ret);

	HysTmobj=clCreateBuffer(context,CL_MEM_READ_WRITE,M*M*sizeof(float),NULL,&ret);
	sHyTmobj=clCreateBuffer(context,CL_MEM_READ_WRITE,M*M*sizeof(float),NULL,&ret);
	ssTmobj=clCreateBuffer(context,CL_MEM_READ_WRITE,M*M*sizeof(float),NULL,&ret);
	ymobj=clCreateBuffer(context,CL_MEM_READ_WRITE,M*sizeof(float),NULL,&ret);
	smobj=clCreateBuffer(context,CL_MEM_READ_WRITE,M*sizeof(float),NULL,&ret);
	Hymobj=clCreateBuffer(context,CL_MEM_READ_WRITE,M*sizeof(float),NULL,&ret);
	sTymobj=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float),NULL,&ret);
	yTHymobj=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float),NULL,&ret);
	
}

void cl_run(){
	b1[0]=fx1(x2[0],x2[1]);
	b1[1]=fx2(x2[0],x2[1]);
	sTy[0]=0;
	yTHy[0]=0;
	for(i=0;i<M;i++){
		Hy[i]=0;
	}
	//Send data to memory buffer
	ret=clEnqueueWriteBuffer(command_queue,Hmobj,CL_TRUE,0,M*M*sizeof(float),H,0,NULL,NULL);
	ret=clEnqueueWriteBuffer(command_queue,H1mobj,CL_TRUE,0,M*M*sizeof(float),H1,0,NULL,NULL);
	ret=clEnqueueWriteBuffer(command_queue,bmobj,CL_TRUE,0,M*sizeof(float),b,0,NULL,NULL);
	ret=clEnqueueWriteBuffer(command_queue,b1mobj,CL_TRUE,0,M*sizeof(float),b1,0,NULL,NULL);
	ret=clEnqueueWriteBuffer(command_queue,x1mobj,CL_TRUE,0,M*sizeof(float),x1,0,NULL,NULL);
	ret=clEnqueueWriteBuffer(command_queue,x2mobj,CL_TRUE,0,M*sizeof(float),x2,0,NULL,NULL);

	ret=clEnqueueWriteBuffer(command_queue,HysTmobj,CL_TRUE,0,M*M*sizeof(float),HysT,0,NULL,NULL);
	ret=clEnqueueWriteBuffer(command_queue,sHyTmobj,CL_TRUE,0,M*M*sizeof(float),sHyT,0,NULL,NULL);
	ret=clEnqueueWriteBuffer(command_queue,ssTmobj,CL_TRUE,0,M*M*sizeof(float),ssT,0,NULL,NULL);
	ret=clEnqueueWriteBuffer(command_queue,ymobj,CL_TRUE,0,M*sizeof(float),y,0,NULL,NULL);
	ret=clEnqueueWriteBuffer(command_queue,smobj,CL_TRUE,0,M*sizeof(float),s,0,NULL,NULL);
	ret=clEnqueueWriteBuffer(command_queue,Hymobj,CL_TRUE,0,M*sizeof(float),Hy,0,NULL,NULL);
	ret=clEnqueueWriteBuffer(command_queue,sTymobj,CL_TRUE,0,sizeof(float),sTy,0,NULL,NULL);
	ret=clEnqueueWriteBuffer(command_queue,yTHymobj,CL_TRUE,0,sizeof(float),yTHy,0,NULL,NULL);

	//Create kerner program with source
	program=clCreateProgramWithSource(context,1,(const char **)&source_str,(const size_t *)&source_size,&ret);
	ret=clBuildProgram(program,1,&device_id,NULL,NULL,NULL);

	//counter 
	tstart = (double)clock()/CLOCKS_PER_SEC;

	//Data parallel with Opencl
	kernel[0]=clCreateKernel(program,"quasinewton1",&ret);
	kernel[1]=clCreateKernel(program,"quasinewton2",&ret);
	kernel[2]=clCreateKernel(program,"quasinewton3",&ret);
	kernel[3]=clCreateKernel(program,"quasinewton4",&ret);

	//Dim Opencl kernel 
	ret=clSetKernelArg(kernel[0],0,sizeof(cl_mem),(void *)&bmobj);
	ret=clSetKernelArg(kernel[0],1,sizeof(cl_mem),(void *)&b1mobj);
	ret=clSetKernelArg(kernel[0],2,sizeof(cl_mem),(void *)&x1mobj);
	ret=clSetKernelArg(kernel[0],3,sizeof(cl_mem),(void *)&x2mobj);
	ret=clSetKernelArg(kernel[0],4,sizeof(cl_mem),(void *)&smobj); 
	ret=clSetKernelArg(kernel[0],5,sizeof(cl_mem),(void *)&ymobj);
	ret=clSetKernelArg(kernel[0],6,sizeof(cl_mem),(void *)&m);

	ret=clSetKernelArg(kernel[1],0,sizeof(cl_mem),(void *)&smobj);
	ret=clSetKernelArg(kernel[1],1,sizeof(cl_mem),(void *)&ymobj);
	ret=clSetKernelArg(kernel[1],2,sizeof(cl_mem),(void *)&Hmobj);
	ret=clSetKernelArg(kernel[1],3,sizeof(cl_mem),(void *)&sTymobj);
	ret=clSetKernelArg(kernel[1],4,sizeof(cl_mem),(void *)&Hymobj);
	ret=clSetKernelArg(kernel[1],5,sizeof(cl_mem),(void *)&ssTmobj);
	ret=clSetKernelArg(kernel[1],6,sizeof(cl_mem),(void *)&m);

	ret=clSetKernelArg(kernel[2],0,sizeof(cl_mem),(void *)&smobj);
	ret=clSetKernelArg(kernel[2],1,sizeof(cl_mem),(void *)&ymobj);
	ret=clSetKernelArg(kernel[2],2,sizeof(cl_mem),(void *)&Hymobj); 
	ret=clSetKernelArg(kernel[2],3,sizeof(cl_mem),(void *)&yTHymobj);
	ret=clSetKernelArg(kernel[2],4,sizeof(cl_mem),(void *)&HysTmobj);
	ret=clSetKernelArg(kernel[2],5,sizeof(cl_mem),(void *)&sHyTmobj);
	ret=clSetKernelArg(kernel[2],6,sizeof(cl_mem),(void *)&m);

	ret=clSetKernelArg(kernel[3],0,sizeof(cl_mem),(void *)&Hmobj);
	ret=clSetKernelArg(kernel[3],1,sizeof(cl_mem),(void *)&H1mobj);
	ret=clSetKernelArg(kernel[3],2,sizeof(cl_mem),(void *)&HysTmobj);
	ret=clSetKernelArg(kernel[3],3,sizeof(cl_mem),(void *)&sHyTmobj);
	ret=clSetKernelArg(kernel[3],4,sizeof(cl_mem),(void *)&sTymobj);
	ret=clSetKernelArg(kernel[3],5,sizeof(cl_mem),(void *)&yTHymobj);
	ret=clSetKernelArg(kernel[3],6,sizeof(cl_mem),(void *)&ssTmobj);
	ret=clSetKernelArg(kernel[3],7,sizeof(cl_mem),(void *)&m);

	//Set Global Work size & local size
	size_t globalWorkSize[2] = {M,M};
	size_t localWorkSize[2] = {M,M};  

	//Run kernel code 
	clEnqueueNDRangeKernel(command_queue, kernel[0], 2, NULL, globalWorkSize, localWorkSize, 0,NULL, NULL);
	clEnqueueNDRangeKernel(command_queue, kernel[1], 2, NULL, globalWorkSize, localWorkSize, 0,NULL, NULL);
	clEnqueueNDRangeKernel(command_queue, kernel[2], 2, NULL, globalWorkSize, localWorkSize, 0,NULL, NULL);
	clEnqueueNDRangeKernel(command_queue, kernel[3], 2, NULL, globalWorkSize, localWorkSize, 0,NULL, NULL);

	//Read result from memory buffer
	ret=clEnqueueReadBuffer(command_queue,H1mobj,CL_TRUE,0,M*M*sizeof(float),H1,0,NULL,NULL);
}

void print_result(){
	printf("loop=%d\n",loop);
	printf("x = %f,y = %f\n",x2[0],x2[1]);
	printf("d[0] = %f,d[1] = %f\n",d[0],d[1]); 
	for(i=0;i<M;i++){
		x1[i]=x2[i];
		b[i]=b1[i];
		for(j=0;j<M;j++){ 
			printf("H1[%d][%d]=%f\n",i,j,H1[i*M+j]);
			H[i*M+j]=H1[i*M+j]; 
		}
	}
}

void opencl_flush(){
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel[0]);
	ret = clReleaseKernel(kernel[1]);
	ret = clReleaseKernel(kernel[2]);
	ret = clReleaseKernel(kernel[3]); 
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(Hmobj);
	ret = clReleaseMemObject(H1mobj);
	ret = clReleaseMemObject(bmobj);
	ret = clReleaseMemObject(b1mobj);
	ret = clReleaseMemObject(x1mobj);
	ret = clReleaseMemObject(x2mobj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(H);
	free(H1);
	free(b);
	free(b1);
	free(x1);
	free(x2);
}



int main(void)
{
	initial();
	cl_start();
	while(flag==0){
		loop++;
		hk();
		alpha_update();
		x2_update();
		flagx();
		cl_run();
		print_result();
		tstop=(double)clock()/CLOCKS_PER_SEC; 
		ttime=tstop-tstart;
		printf("time=%f\n\n",ttime); 
	}
	printf("DONE\n");
	opencl_flush();
}