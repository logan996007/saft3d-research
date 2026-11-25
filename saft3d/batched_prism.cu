// Filled at runtime with the complete 6-choose-4 catalogue and oriented face masks.
__constant__ int V[15][4]=CANDIDATE_TABLE;
__constant__ unsigned M[15][2]=MASK_TABLE;
extern "C" __global__ void batch_front(
    const double *p,const long long *cap,const long long *level,
    const unsigned *signs,const double *area2,const double *height,
    long long nf,long long nv,long long total,long long *out,unsigned *failed){
    const long long w=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(w>=total)return;
    const long long f=w%nf,k=w/nf;
    const double expected=area2[f]*(height[k+1]-height[k]);
    long long ids[6];double x[6][3];
    for(int j=0;j<6;j++){
        ids[j]=level[(k+j/3)*nv+cap[3*f+j%3]];
        for(int a=0;a<3;a++)x[j][a]=p[3*ids[j]+a];
    }
    unsigned pos=signs[2*f],neg=signs[2*f+1],initial=pos|neg,eligible=0u;
    double det[15],sum=0.;
    for(int j=0;j<15;j++){
        double u[3],v[3],s[3];
        for(int a=0;a<3;a++){
            double origin=x[V[j][0]][a];
            u[a]=x[V[j][1]][a]-origin;v[a]=x[V[j][2]][a]-origin;s[a]=x[V[j][3]][a]-origin;
        }
        det[j]=(u[1]*v[2]-u[2]*v[1])*s[0]+(u[2]*v[0]-u[0]*v[2])*s[1]
              +(u[0]*v[1]-u[1]*v[0])*s[2];
        unsigned wall=(M[j][0]|M[j][1])&EXTERIOR_BITS;
        if(fabs(det[j]-expected)<=1.e-10*expected && !(wall&~initial))
            eligible|=1u<<j;
    }
    for(int step=0;step<3;step++){
        unsigned active=pos|neg;int choice=-1;
        for(int j=0;j<15;j++){
            unsigned a=M[j][0],b=M[j][1],wall=(a|b)&EXTERIOR_BITS;
            bool legal=(eligible&(1u<<j)) && ((a&pos)|(b&neg))
                      && !((a&neg)|(b&pos)) && !(wall&~active);
            if(legal && choice<0)choice=j;
        }
        if(choice<0){atomicOr(failed,1u);return;}
        unsigned a=M[choice][0],b=M[choice][1],mask=a|b;
        pos=(pos&~mask)|(b&~active);neg=(neg&~mask)|(a&~active);
        eligible &= ~(1u<<choice);sum+=det[choice];
        for(int j=0;j<4;j++)out[12*w+4*step+j]=ids[V[choice][j]];
    }
    if((pos|neg) || fabs(sum-3.*expected)>3.e-10*expected)atomicOr(failed,2u);
}
