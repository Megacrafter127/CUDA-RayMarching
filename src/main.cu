/*
 * main.cu
 *
 *  Created on: 31.07.2019
 *      Author: Megacrafter127
 */

#include <raymarching.cuh>

#include <cmath>
#include <ctime>

#include <SDL2/SDL_mouse.h>

#define CHUNK_SIZE 16

#define IMG_CHUNKS_WIDTH 50
#define IMG_WIDTH (IMG_CHUNKS_WIDTH*CHUNK_SIZE)
#define IMG_CHUNKS_HEIGHT 40
#define IMG_HEIGHT (IMG_CHUNKS_HEIGHT*CHUNK_SIZE)

#if IMG_WIDTH>IMG_HEIGHT
#define IMG_MIN IMG_HEIGHT
#define IMG_MAX IMG_WIDTH
#else
#define IMG_MAX IMG_HEIGHT
#define IMG_MIN IMG_WIDTH
#endif

static SDL_Surface *surf;

__managed__ static world_t world;

const static clock_t start=clock();

__constant__ static scalarType time_f=.0f;

inline static void updateTime() {
	scalarType t=static_cast<scalarType>(clock()-start)/CLOCKS_PER_SEC;
	cudaMemcpyToSymbol(time_f,&t,sizeof(scalarType),cudaMemcpyHostToDevice);
}

#define SHAPE_COUNT 9

__constant__ static shape_t shapes[SHAPE_COUNT];

typedef struct {
	floatColor_t color;
} simpleColor_t;

typedef struct : simpleColor_t {
	float3 center;
	float radius;
} sphereData_t;

__constant__ static sphereData_t shapeData[SHAPE_COUNT];

__device__ static float sphereDistance(const void *shapeData, vectorType point, size_t frame) {
	register const sphereData_t * const sphere=static_cast<const sphereData_t*>(shapeData);
	register vectorType diff=point+-1*sphere->center;
	return norm(diff)-sphere->radius;
}
__managed__ distanceFunction sphereDistAddr=sphereDistance;

__device__ static float cubeDistance(const void *shapeData, vectorType point, size_t frame) {
	register const sphereData_t * const sphere=static_cast<const sphereData_t*>(shapeData);
	register vectorType diff=point+-1*sphere->center;
	diff.x=fabs(diff.x);
	diff.y=fabs(diff.y);
	diff.z=fabs(diff.z);
	if(diff.y>diff.x) diff.x=diff.y;
	if(diff.z>diff.x) diff.x=diff.z;
	return diff.x-sphere->radius;
}
__managed__ distanceFunction cubeDistAddr=cubeDistance;

__device__ static floatColor_t glowColor(const void *shapeData, vectorType point, scalarType distance, scalarType divergence, size_t frame, size_t steps) {
	register floatColor_t color=static_cast<const simpleColor_t*>(shapeData)->color;
	const register scalarType stepf=1-1/(steps*.0625f+1),divf=1-distance/divergence;
	color.r*=divf*stepf;
	color.g*=divf*stepf;
	color.b*=divf*stepf;
	return color;
}
__managed__ colorFunc glowColorAddr=glowColor;

int handleError(cudaError_t err) {
	if(err==cudaSuccess) return 0;
	fprintf(stderr,"%s: %s\n",cudaGetErrorName(err),cudaGetErrorString(err));
	return 1;
}

typedef struct {
	vectorType face,dx,dy;
} camplane_t;

__constant__ camplane_t camplane;
static camplane_t cam;
static float2 mouseMotion;
static scalarType mouseSensitivity=.5f;
static scalarType movementSpeed=1;
static unsigned char keymask=0;
#define KEY_FORWARD 0
#define KEY_BACKWARD 1
#define KEY_LEFT 2
#define KEY_RIGHT 3
#define KEY_UP 4
#define KEY_DOWN 5
#define MASK(bit) (1<<(bit))
#define TEST(mask,bit) ((mask>>bit)&1)
#define TESTAB(mask,p,n) (TEST(mask,p)-TEST(mask,n))
static scalarType fov=-1.0f/(IMG_MAX-1);
static scalarType fovSpeed=fov*.09375f;
static const vectorType vecx=make_float3(1,0,0);
static const vectorType vecy=make_float3(0,1,0);
static const vectorType vecz=make_float3(0,0,1);


__device__ vectorType raydir(scalarType &divergence, int3 pos, size_t frame) {
	register vectorType ret=camplane.face+pos.x*camplane.dx+pos.y*camplane.dy;
	vectorType delta[4]={ret+camplane.dx,ret+-camplane.dx,ret+camplane.dy,ret+-camplane.dy};
	ret=ret/norm(ret);
	divergence=INFINITY;
	for(int i=0;i<4;i++) {
		delta[i]=-delta[i]/norm(delta[i]);
		register scalarType ldiv=norm(ret+delta[i]);
		if(ldiv<divergence) divergence=ldiv;
	}
	divergence=(norm(camplane.dx)+norm(camplane.dy))/norm(ret);
	return ret;
}
__managed__ rayFunc rf=raydir;
static clock_t last=clock();
static int postFrame(size_t frame, void *data) {
	updateTime();
	const clock_t t=clock();
	register vectorType delta;
	delta+=normv(cam.face)*TESTAB(keymask,KEY_FORWARD,KEY_BACKWARD);
	delta+=normv(cam.dx)*TESTAB(keymask,KEY_RIGHT,KEY_LEFT);
	delta+=normv(cam.dy)*TESTAB(keymask,KEY_DOWN,KEY_UP);
	world.camera.pos+=(movementSpeed*(t-last)/CLOCKS_PER_SEC)*delta;
	const register scalarType cx=coss(mouseMotion.x),sx=sins(mouseMotion.x);
	const register scalarType cy=coss(mouseMotion.y),sy=sins(-mouseMotion.y);
	cam.dx=fov*make_float3(cx,0,sx);
	cam.dy=fov*make_float3(sx*sy,cy,-cx*sy);
	cam.face=make_float3(-sx*cy,sy,cx*cy);

	cudaMemcpyToSymbol(camplane,&cam,sizeof(camplane_t));
	last=t;
	return 0;
}

static int event(SDL_Event *event, void *data) {
	int key=-1;
	switch(event->type) {
	case SDL_KEYDOWN:
	case SDL_KEYUP:
		switch(event->key.keysym.scancode) {
		case SDL_SCANCODE_ESCAPE:
			return 2;
		case SDL_SCANCODE_W:
			key=KEY_FORWARD;
			break;
		case SDL_SCANCODE_A:
			key=KEY_LEFT;
			break;
		case SDL_SCANCODE_S:
			key=KEY_BACKWARD;
			break;
		case SDL_SCANCODE_D:
			key=KEY_RIGHT;
			break;
		case SDL_SCANCODE_SPACE:
			key=KEY_UP;
			break;
		case SDL_SCANCODE_LSHIFT:
			key=KEY_DOWN;
			break;
		}
		if(key!=-1) {
			key=MASK(key);
			switch(event->key.state) {
			case SDL_PRESSED:
				keymask|=key;
				break;
			case SDL_RELEASED:
				keymask&=~key;
			}
		}
		break;
	case SDL_MOUSEMOTION:
		mouseMotion.x-=mouseSensitivity*fov*event->motion.xrel;
		mouseMotion.y-=mouseSensitivity*fov*event->motion.yrel;
		break;
	case SDL_MOUSEWHEEL:
		fov+=fovSpeed*(event->wheel.x+event->wheel.y);
		break;
	}
	return defaultEventFunction(event,data);
}

const static float xs[SHAPE_COUNT]={ 10, 10,  0,-10,-10,-10,  0, 10};

int main() {
	cam.dy=fov*vecy;
	cam.dx=fov*vecx;
	cam.face=vecz;
	cudaMemcpyToSymbol(camplane,&cam,sizeof(camplane_t));
	world.camera.rays=rf;
	world.camera.pos=-2*vecz;
	postFrame(0,NULL);
	cudaGetSymbolAddress((void**)&(world.shapes),shapes);
	world.shapeCount=SHAPE_COUNT;
	world.maxErr=0.0625f;

	sphereData_t *sd;
	cudaGetSymbolAddress((void**)&sd,shapeData);

	for(int i=0;i<SHAPE_COUNT;i++,sd++) {
		cudaMemcpyToSymbol(shapes,&glowColorAddr,sizeof(colorFunc),sizeof(shape_t)*i+offsetof(shape_t,colorFunction));
		cudaMemcpyToSymbol(shapes,i%2?&sphereDistAddr:&cubeDistAddr,sizeof(distanceFunction),sizeof(shape_t)*i+offsetof(shape_t,distanceFunc));
		cudaMemcpyToSymbol(shapes,&sd,sizeof(void*),sizeof(shape_t)*i+offsetof(shape_t,shapeData));
	}
	sphereData_t sample;

	sample.center=make_float3(0,0,0);
	sample.color.a = 1;
	sample.color.r = .75f;
	sample.color.g = 0;
	sample.color.b = .75f;
	sample.radius = .0625f;
	cudaMemcpyToSymbol(shapeData,&sample,sizeof(sphereData_t));

	for(size_t i=0;i<SHAPE_COUNT-1;i++) {
		sample.center=make_float3(xs[i],0,xs[(i+2)%(SHAPE_COUNT-1)]);
		sample.color.a = 1;
		sample.color.r = i%2?0:1;
		sample.color.g = (i/2)%2?0:1;
		sample.color.b = (i/4)%2?0:1;
		if(i%8==7) {
			sample.color.r=.125f;
			sample.color.g=.125f;
			sample.color.b=.125f;
		}
		sample.radius = i%2*3+i%3+1;
		cudaMemcpyToSymbol(shapeData,&sample,sizeof(sphereData_t),sizeof(sphereData_t)*(i+1));
	}
	surf=createFullscreenSurface("Raymarching Test",true,true);
	SDL_DisplayMode dm;
	float delay=0;
	if(SDL_GetCurrentDisplayMode(0,&dm)==0) {
		delay=1.0f/dm.refresh_rate;
	}
	SDL_SetRelativeMouseMode(SDL_TRUE);
	int ret=handleError(autoRenderShapes(surf,&world,delay,postFrame,NULL,event));
	destroySurface(surf);
	return ret;
}
