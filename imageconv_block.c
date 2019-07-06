#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<omp.h>

typedef struct {
  unsigned char red,green,blue;
} PPMPixel;

typedef struct {
  int x, y;
  PPMPixel *data;
} PPMImage;

typedef struct {
  unsigned char gs;
} PPMPixelGS;


typedef struct {
  int x, y;
  PPMPixelGS *data;
} PPMImageGS;



#define RGB_COMPONENT_COLOR 255


void writePPMGS(const char *filename, PPMImageGS *img)
{
  FILE *fp;
  //open file for output
  fp = fopen(filename, "wb");
  if (!fp) {
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }

  //write the header file
  //image format
  fprintf(fp, "P5\n");

    

  //image size
  fprintf(fp, "%d %d\n",img->x,img->y);

  // rgb component depth
  fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

  // pixel data
  fwrite(img->data, img->x, img->y, fp);
  fclose(fp);
}

int sharpen[3][3] = {{0,-1,0},{-1,5,-1},{0,-1,0}};
int blur[3][3] = {{0,1,0},{1,4,1},{0,1,0}};
int emboss[3][3] = {{-2,-1,0},{-1,1,1},{0,-1,2}};
int emboss5[5][5] = {{-4,-3,-2,-1,0},{-3,-2,-1,0,1},{-2,-1,1,1,2},{-1,0,-1,2,3},{0,1,2,3,4}};
int emboss7[7][7] = {{-6,-5,-4,-3,-2,-1,0},{-5,-4,-3,-2,-1,0,1},{-4,-3,-2,-1,0,1,2},{-3,-2,-1,1,1,2,3},{-2,-1,0,-1,2,3,4},{-1,0,1,2,3,4,5},{0,1,2,3,4,5,6}};
int blur5[5][5] = {{0,0,1,0,0},{0,1,1,1,0},{1,1,1,1,1},{0,1,1,1,0},{0,0,1,0,0}};
int blur7[7][7] = {{0,0,0,1,0,0,0},{0,0,1,1,1,0,0},{0,1,1,1,1,1,0},{1,1,1,1,1,1,1},{0,1,1,1,1,1,0},{0,0,1,1,1,0,0},{0,0,0,1,0,0,0}};
int sobel[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
int sobel5[5][5] = {{2,1,0,-1,-2},{3,2,0,-2,-3},{4,3,0,-3,-4},{3,2,0,-2,-3},{2,1,0,-1,-2}};
int sobel7[7][7] = {{3,2,1,0,-1,-2,-3},{4,3,2,0,-2,-3,-4},{5,4,3,0,-3,-4,-5},{6,5,4,0,-4,-5,-6},{5,4,3,0,-3,-4,-5},{4,3,2,0,-2,-3,-4},{3,2,1,0,-1,-2,-3}};

PPMImage * convolution_wrap(PPMImage * im,int kernel[3][3],int h){
	float tim=omp_get_wtime();
	int sum=0;
	int rows = im->x;
  	int cols = im->y;
	int i,j,k,l,r,g,b,m;
	for(i=0;i<2*h+1;i++)
		for(j=0;j<2*h+1;j++) sum+=kernel[i][j];  //computing sum of the kernel
	if(sum==0) sum=1; // sum exception
	PPMImage *im2 = (PPMImage *) malloc(sizeof(PPMImage));  //new image for result
	im2->x = rows;  
  	im2->y = cols;  
  	im2->data = (PPMPixel *) malloc(rows*cols*sizeof(PPMPixel));
  	int fac[30],a,c,pt;
  	for(i=0;i<30;i++) fac[i]=0; // fac array to store the factors of p
  	int p;
  	#pragma omp parallel shared(pt,p)  // computing factors
  	{
  		int id=omp_get_thread_num();
  		p=omp_get_num_threads();
  		if(id==0){
  			pt=0;
			for(c=2;c<p;c++){
				if(p%c==0){
					fac[pt]=c;
					pt++;

				}
			}
		}
	}
	tim=omp_get_wtime()-tim;
  	{
		for(m=0;m<(pt+1)/2;m++){ //iterating through half of the factors
			a=fac[m];
			if(a==0)
			break;
			c=p/fac[m];  // to get the opposite factor
			double time = omp_get_wtime();
			#pragma omp parallel private(i,j,k,l,r,g,b) firstprivate(rows,cols,a,c,im2,im,sum) //parallel section
			{
				int id=omp_get_thread_num();
				int start_j = (id % a) * (rows / a);	//dividing the work using ID
				int end_j = ((id+1) % a)  * (rows / a);	//partitioning the blocks
		
				int start_i = (id / c) * (cols / c);		//	0|1|2
				int end_i = ((id / c)+1) * (cols / c);		//	3|4|5		id p=9 then this will be the distribution
				if((id + 1) % a == 0)						//	6|7|8
					end_j = rows;
		
				for(i=start_i;i< end_i;i++){
					for(j=start_j;j<end_j;j++){ // iterating through the blocks
						int idx=rows*i+j;
				  		PPMPixel *temp2 = im2->data + idx;
						int cr=0,cb=0,cg=0; // for storing the sum
						for(k=i-h;k<=i+h;k++){
							if(k<0)	k+=rows;	//boundary condition for rows
							if(k>=rows) k-=rows;
							for(l=j-h;l<=j+h;l++){
								if(l<0) l+=cols;	// boundary condition for cols
								if(l>=cols) l-=cols; 
								{
								idx=rows*k+l;
								PPMPixel *temp = im->data + idx;
								r = temp->red;
				  				g = temp->green;
				  				b = temp->blue;
								cr+=kernel[k-i+h][l-j+h]*r; //convolution steps
								cg+=kernel[k-i+h][l-j+h]*g;
								cb+=kernel[k-i+h][l-j+h]*b;			
							}}
						}
						temp2->red= (cr/sum>255.0) ? 255.0 : ((cr/sum<0.0) ? 0.0 : cr/sum) ;	// assignment
						temp2->green = (cg/sum>255.0) ? 255.0 : ((cg/sum<0.0) ? 0.0 : cg/sum);
						temp2->blue = (cb/sum>255.0) ? 255.0 : ((cb/sum<0.0) ? 0.0 : cb/sum);
					}
				}
			}
			time=omp_get_wtime()-time;
			time+=tim;
			printf("Configuration a: %d, b: %d and time: %lf\n",a,c,time);
		}
  	}
	return im2;
}

static PPMImage *readPPM(const char *filename)
{
  char buff[16];
  PPMImage *img;
  FILE *fp;
  int c, rgb_comp_color;
  //open PPM file for reading
  fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }

  //read image format
  if (!fgets(buff, sizeof(buff), fp)) {
    perror(filename);
    exit(1);
  }

  //check the image format
  if (buff[0] != 'P' || buff[1] != '6') {
    fprintf(stderr, "Invalid image format (must be 'P6')\n");
    exit(1);
  }

  //alloc memory form image
  img = (PPMImage *)malloc(sizeof(PPMImage));
  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  //check for comments
  c = getc(fp);
  while (c == '#') {
    while (getc(fp) != '\n') ;
    c = getc(fp);
  }

  ungetc(c, fp);
  //read image size information
  if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
    fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
    exit(1);
  }

  //read rgb component
  if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
    fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
    exit(1);
  }

  //check rgb component depth
  if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
    fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
    exit(1);
  }

  while (fgetc(fp) != '\n') ;
  //memory allocation for pixel data
  img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  //read pixel data from file
  if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
    fprintf(stderr, "Error loading image '%s'\n", filename);
    exit(1);
  }

  fclose(fp);
  return img;
}

void writePPM(const char *filename, PPMImage *img)
{
  FILE *fp;
  //open file for output
  fp = fopen(filename, "wb");
  if (!fp) {
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }

  //write the header file
  //image format
  fprintf(fp, "P6\n");

  //comments


  //image size
  fprintf(fp, "%d %d\n",img->x,img->y);

  // rgb component depth
  fprintf(fp, "%d\n",255);

  // pixel data
  fwrite(img->data, 3 * img->x, img->y, fp);
  fclose(fp);
}


int main(int argc, char* argv[]){
  /*if(argc<2)
    printf("Enter filename\n");
  char* filename=argv[1];*/
  int i=0,n;
  char *filename;
  for(i=0;i<6;i++){
	  	if(i==0) filename = "lena-1024.ppm";
	  	else if(i==1) filename = "lena-1080.ppm";
	  	else if(i==2) filename = "lena-1200.ppm";
	  	else if(i==3) filename = "lena-1252.ppm";
	  	else if(i==4) filename = "lena-1920.ppm";
	  	else if(i==5) filename = "lena-2048.ppm";
		
	  PPMImage *image;
	  clock_t start, end;
	
	  image = readPPM(filename);
	
	  //start=clock();
	  /* float start_omp=omp_get_wtime(); */
	  PPMImage * x = convolution_wrap(image,sobel,1);		//call the function using different kernels
	  /* float stop_omp=omp_get_wtime(); */
	  //end=clock();
	    
	  //printf("Time: %0.10f\n", (end-start)/(double)CLOCKS_PER_SEC );
	  /* printf("Time: %0.10f\n", (stop_omp-start_omp) ); */
	  if(i==0) writePPM("lenags-1024.ppm",x);
	  else if(i==1) writePPM("lenags-1080.ppm",x);
	  else if(i==2) writePPM("lenags-1200.ppm",x);
	  else if(i==3) writePPM("lenags-1252.ppm",x);
	  else if(i==4) writePPM("lenags-1920.ppm",x);
	  else if(i==5) writePPM("lenags-2048.ppm",x);
	}

return 0;
}


