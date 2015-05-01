__kernel void quasinewton1(__global float* b,__global float* b1,__global float* x1,__global float* x2,__global float* s,__global float* y,int m)
{
    int i = get_global_id(0);
	s[i]=x2[i]-x1[i];
	y[i]=b1[i]-b[i];
}

__kernel void quasinewton2(__global float* s,__global float* y,__global float* H,__global float* sTy,__global float* Hy,__global float* ssT,int m)
{
	int i = get_global_id(0);
	int j;
	for(j=0;j<m;j++){
		sTy[0]+=s[j]*y[j];
		Hy[i]+=H[i*m+j]*y[j];
		ssT[i*m+j]=s[i]*s[j];
	}
}

__kernel void quasinewton3(__global float* s,__global float* y,__global float* Hy,__global float* yTHy,__global float* HysT,__global float* sHyT,int m)
{
	int i = get_global_id(0);
	int j;
	for(j=0;j<m;j++){
		yTHy[0]+=y[j]*Hy[j];
		HysT[i*m+j]=Hy[i]*s[j];
		sHyT[i*m+j]=s[i]*Hy[j];
	}
}

__kernel void quasinewton4(__global float* H,__global float* H1,__global float* HysT,__global float* sHyT,__global float* sTy,__global float* yTHy,__global float* ssT,int m)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	H1[i*m+j]=H[i*m+j]-((HysT[i*m+j]+sHyT[i*m+j])/sTy[0])+(1+(yTHy[0]/sTy[0]))*(ssT[i*m+j]/sTy[0]);
}

