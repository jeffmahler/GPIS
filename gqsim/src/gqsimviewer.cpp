#ifdef _WIN32
#include <windows.h>
#endif

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <iostream>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <time.h>
#include "world.h"
#include "graspset.h"
#include "glsub.h"
#include "utilfunc.h"

#ifdef _WIN32
#include <commdlg.h>		// OPENFILENAME, GetOpenFileName()
static OPENFILENAME ofn;    
static OPENFILENAME ofn2;    
#include <Shlobj.h>			// SHBrowseForFolder()
static BROWSEINFO bInfo;
static int CALLBACK BrowseCallbackProc(HWND hwnd,UINT uMsg, LPARAM lParam, LPARAM lpData)
{
	// If the BFFM_INITIALIZED message is received
	// set the path to the start path.
	switch (uMsg)
	{
		case BFFM_INITIALIZED:
		{
			if (NULL != lpData)
			{
				SendMessage(hwnd, BFFM_SETSELECTION, TRUE, lpData);
			}
		}
	}

	return 0; // The function should always return 0.
}
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

using namespace std;

#define KEY_SPACE 32
#define KEY_ESC 27
#define KEY_ENTER 13

// system color
GLfloat textcolor[] = {0,0,0,1};
GLfloat backgroundcolor[] = {1,1,1,1};
GLfloat lightpos0[] = {0,0,1,1};
GLfloat lightpos1[] = {1,1,1,1};
GLfloat lightpos2[] = {1,-1,1,1};
GLfloat lightpos3[] = {-1,1,1,1};
GLfloat lightpos4[] = {-1,-1,1,1};

// mode
bool bsimul=false;
bool breplay=false; 
bool bctrlpressed = false, bshiftpressed = false, baltpressed = false; // cannot handle multiple key pressing at the same time

// world
World *pworld = NULL;
float mscale = 1; // model scale (this will be set later to be the radius of the estimated bounding shpere of the character)
bool bshowboundingbox = false;
bool bshowcontactforces = false, bshowcontactsurfacepatch = false;
bool bshowcheckerboard = true;
int rendermode = 0;

// grasp set
GraspSet graspset;
int current_grasp_idx=-1;

// checkerboard
static float light_gray[3] = {0.9, 0.9, 0.9};
static float dark_gray[3] = {0.7, 0.7, 0.7};
static float ground_position[3] = {0, 0, 0};
static float ground_normal[3] = {0, 0, 1};

// simulation
double current_time = 0; // this is also used for replay
double max_simul_time = 60;
int max_counter_save = int(1./(double(120)*0.001));
int counter_save=0;
bool bshowmotormsg = false;
double start_simul_time=0, end_simul_time=0;
bool bdatasavecontact;

// replay simulated motion
int current_frame_idx=0; // current replay frame index
const int replay_speed_set_size = 16;
double replay_speed_set[replay_speed_set_size] = {0.001, 0.005, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 100.0};
int replay_speed_cnt = 7;
double replay_speed = replay_speed_set[replay_speed_cnt]; // replay speed
double base_render_refersh_rate = 30; // 30 Hz
double current_render_refersh_rate = base_render_refersh_rate; 
int num_skip_frames = 0; // number of frames to be skipped when replay
double time_loading_rendering = 0;

// viewer state
Vec3 spoint_prev;
SE3 mvmatrix, mvmatrix_prev; // view change
double &xcam = mvmatrix[12], &ycam = mvmatrix[13]; // view translate
double &sdepth = mvmatrix[14]; // zoom in/out
float fovy=64.0f, zNear=0.001f, zFar=10000.0f;
float aspect = 5.0f/4.0f;
long winsizeX, winsizeY;
int downX, downY;
bool bRotateView = false, bTranslateView = false, bZoomInOut = false, bSliderControl = false, bShowUsage = false;
GLfloat light0Position[] = {1, 1, 1, 1};
bool bshowusage = false;
bool bshowallgrasps = false;
bool bshowfingerprint = false;
bool bshowtextinfo = true;
double alpha_hand_for_rendering_all_grasps = 0.3;
int current_fingerprint_idx = -1;
int fp_ref_hit_cnt_for_rendering = 1000;
float fp_ref_hit_cnt_ratio_init = 0.3;

// slider control position
int sliderXi=20, sliderXf=520, sliderX=sliderXi;
int sliderY=10; // vertical position (from the bottom line of the window)

// declaration
void StepSimulation(void);
void StopSimulation(void);
void StartSimulation(void);
void ReadSimulData(void);
void ApplyGrasp(int idx);

// test function
static GLUquadricObj *qobj = gluNewQuadric();
static Vec3 _p(0,0,0), _x(0,0,0);
static int _vidx = -1;

string getManual(void)
{
	stringstream ss; 
	ss << "-------------------" << endl;
	ss << "   gqsim manual    " << endl;
	ss << "-------------------" << endl;
	ss << " ESC: exit program" << endl;
	ss << " ENTER: start/pause simulation" << endl;
	ss << " SPACE: play/pause replay" << endl;
	ss << " b: show/hide bounding boxes" << endl;
	ss << " c: show/hide contact forces" << endl;
	ss << " c+ctrl: show/hide contact surface patch" << endl;
	ss << " c+shift: show/hide grasp contact center after grasping" << endl;
	ss << " d: change rendering mode (original/wire/collision/all)" << endl;
	ss << " e+ctrl: error tracking analysis for the current grasp" << endl;
	ss << " f: save force data to fdata.m" << endl;
	ss << " f+ctrl: calculate and save finger print of the current grasp set" << endl;
	ss << " g: show/hide ground" << endl;
	ss << " h: show/hide hand or object" << endl;
	ss << " i: show model information" << endl;
	ss << " i+ctrl: show/hide motor message during simulation" << endl;
	ss << " k: remove the current grasp" << endl;
	ss << " k+ctrl: remove all grasps colliding with ground" << endl;
	ss << " m+ctrl: measure grasp quality of the grasps with considering uncertainty" << endl;
	ss << " m+shift: measure grasp quality of the grasps with considering uncertainty (without simulation data saving)" << endl;
	ss << " m+alt: measure grasp quality of the grasps without considering uncertainty" << endl;
	ss << " n: show/hide face/vertex normals of the object" << endl;
	ss << " n+ctrl: show/hide face/vertex normals of the hand geometry" << endl;
	ss << " o: open simulation data file(s)" << endl;
	ss << " o+ctrl: open grasp set file" << endl;
	ss << " o+shift: open model xml file" << endl;
	ss << " o+alt: open state file" << endl;
	ss << " p+ctrl: pick farthest 30 grasps" << endl;
	ss << " r: reset time and prepare a new simulation" << endl;
	ss << " s: save simulation data to file" << endl;
	ss << " s+ctrl: save grasp set to file" << endl;
	ss << " s+alt: save current state to file" << endl;
	ss << " t: test function" << endl;
	ss << " u: show/hide usage on the window" << endl;
	ss << " w: remove duplicate grasps" << endl;
	ss << " x: no text info on the screen" << endl;
	ss << " y: show/hide coordinate frames" << endl;
	ss << " left: step backward (replay)" << endl;
	ss << " left+ctrl: previous grasp" << endl;
	ss << " right: step forward (replay)" << endl;
	ss << " right+ctrl: next grasp" << endl;
	ss << " down: decrease replay speed" << endl;
	ss << " down+ctrl: decrease base refersh rate for rendering" << endl;
	ss << " down+shift: scale down force/moment for rendering" << endl;
	ss << " down+alt: decrease alpha when rendering all grasps" << endl;
	ss << " up: increase replay speed" << endl;
	ss << " up+ctrl: increase base refersh rate for rendering" << endl;
	ss << " up+shift: scale up force/moment for rendering" << endl;
	ss << " up+alt: increase alpha when rendering all grasps" << endl;
	ss << " F1: show manual (terminal)" << endl;
	ss << " F2: restore contact saving option" << endl;
	ss << " F3: sort the current grasp set" << endl;
	ss << " F4: shuffle the current grasp set" << endl;
	ss << " F5: save new grasp set specified by a grasp index file" << endl;
	ss << " F6: save link and object geometries to obj files" << endl;
	ss << " F7: show/hide all grasps (with alpha rendering)" << endl;
	ss << " F8: show/hide finger print" << endl;
	ss << "-------------------" << endl;
	return ss.str();
}


void PlotString(void *font, int x, int y, const char *string)
{        
	// switch to projection mode	
	glMatrixMode(GL_PROJECTION);	
	glPushMatrix();	
	// reset matrix	
	glLoadIdentity();	
	//Set the viewport	
	glViewport(0, 0, winsizeX, winsizeY);
	// set a 2D orthographic projection	
	glOrtho(0, winsizeX, winsizeY, 0, -100, 100);	
	glMatrixMode(GL_MODELVIEW);	
	glPushMatrix();	
	glLoadIdentity();
	glTranslatef(0.0f,0.0f,99.9f);
	// draw
	glRasterPos2i(x,y);
	// glutBitmapString( font, (const unsigned char *)string ); // this doesn't work for OS X
	for (const char *c = string; *c != '\0'; c++) {
		glutBitmapCharacter(font, *c);
	}
	// restore
	glMatrixMode(GL_PROJECTION);	
	glPopMatrix ();	
	glMatrixMode(GL_MODELVIEW);	
	glPopMatrix ();
}  

void DrawTextInfo(void)
{
	if ( !bshowtextinfo ) return;

	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 15.0f); // range 0 ~ 128
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, textcolor);
	//glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glColor4fv(textcolor);

	char *str = new char[200]; 
	int posy = 20;
	if ( !!pworld ) {
		// simulation info
		if ( bsimul ) {
			sprintf(str, "t = %7.4f", current_time);
		} else if ( pworld->_simuldata.size() > 0 ) {
			sprintf(str, "t = %7.4f, tf = %7.4f, replay = %fx, skip frames = %d, refresh rate = %d(Hz)", current_time, pworld->_simuldata.m_list_time.back(), replay_speed, num_skip_frames, int(current_render_refersh_rate));
		} else {
			sprintf(str, "t = %7.4f", current_time);
		}
		PlotString(GLUT_BITMAP_HELVETICA_12, 20, posy, str);
		posy += 15;

		// grasp set info
		if ( graspset.size() > 0 ) {
			if ( current_grasp_idx >= 0 && current_grasp_idx < graspset.size() ) {
				sprintf(str, "current grasp index = %d, number of grasps loaded = %d, quality score = %f", current_grasp_idx, graspset.size(), graspset._grasps[current_grasp_idx]._score);
			} else {
				sprintf(str, "num of grasps loaded = %d", graspset.size());
			}
			PlotString(GLUT_BITMAP_HELVETICA_12, 20, posy, str);
			posy += 15;
		}
		if ( bshowallgrasps ) {
			sprintf(str, "alpha for hand rendering = %f", alpha_hand_for_rendering_all_grasps); 
			PlotString(GLUT_BITMAP_HELVETICA_12, 20, posy, str);
			posy += 15;
		}
		if ( bshowfingerprint ) {
			if ( current_fingerprint_idx >= 0 && current_fingerprint_idx < pworld->_fingerprinthistory.size() ) {
				sprintf(str, "current fingerprint index = %d, ref cnt = %d, number of fingerprints = %d, ref cnt for coloring = %d", current_fingerprint_idx, pworld->_fingerprinthistory.getFingerPrint(current_fingerprint_idx)._hit_cnt_ref, pworld->_fingerprinthistory.size(), fp_ref_hit_cnt_for_rendering);
			} else {
				sprintf(str, "number of fingerprints = %d", pworld->_fingerprinthistory.size());
			}
			PlotString(GLUT_BITMAP_HELVETICA_12, 20, posy, str);
			posy += 15;
		}
	}

	// draw a slider
	int h = (sliderXf-sliderXi)/10;
	for (int x=sliderXi; x<sliderXf; x+=h) {
		PlotString(GLUT_BITMAP_HELVETICA_12, x, winsizeY-sliderY, "|------");
	}
	PlotString(GLUT_BITMAP_HELVETICA_12, sliderXf, winsizeY-sliderY, "|");
	PlotString(GLUT_BITMAP_HELVETICA_12, sliderX, winsizeY-sliderY, "[^]");
	sprintf(str, "(%d%%)", 100*(sliderX-sliderXi)/(sliderXf-sliderXi));
	PlotString(GLUT_BITMAP_HELVETICA_12, sliderXf+15, winsizeY-sliderY, str);

	// usage
	if ( bshowusage ) {
		char str2[] = "ESC : exit, u : show/hide usage, F1 : show full manual (terminal)\nshift+o : open model, o : open simulation file, ctrl+o : open grasp set file\nENTER : start/pause simulation, SPACE : start/pause replay, r : init time and simulation, \nup/down : replay speed up/down, left/right : one-step bwd/fwd, ctrl+left/right : navigate grasps\nc: show/hide contact forces, d : change rendering mode, shift+up/down : scale forces";
		PlotString(GLUT_BITMAP_HELVETICA_12, 20, winsizeY-75-15, str2);
	} else {
		if ( !!pworld ) {
			char str2[] = "ESC : exit, u : show/hide usage, F1 : show full manual (terminal)";
			PlotString(GLUT_BITMAP_HELVETICA_12, 20, winsizeY-15-15, str2);
		} else {
			char str2[] = "ESC : exit, u : show/hide usage, shift+o : open model";
			PlotString(GLUT_BITMAP_HELVETICA_12, 20, winsizeY-15-15, str2);
		}
	}
}

void ShowModelInfo(void)
{
	if ( !pworld ) return;
	for (size_t i=0; i<pworld->_phand->_motors.size(); i++) {
		cout << pworld->_phand->_motors[i] << endl;
	}
	if ( !!pworld->getObject() ) {
		cout << "object: " << pworld->getObject()->getName() << endl;
		cout << "  mass = " << pworld->getObject()->getMass() << endl;
		cout << "  I = " << pworld->getObject()->_I;
		Vec3 p = pworld->getObject()->_T.GetPosition();
		SO3 R = pworld->getObject()->_T.GetRotation();
		Vec3 w = Log(R); double ang = w.Normalize();
		cout << "  translation = " << p[0] << " " << p[1] << " " << p[2] << endl;
		cout << "  axisrotation = " << w[0] << " " << w[1] << " " << w[2] << " " << ang*180./3.14159 << endl;
	}
	cout << "number of collision surface pairs = " << pworld->_colchecker.getNumCollisionSurfacePairs() << endl;
}

void DrawModel(void) 
{
	if ( !pworld ) return;
	if ( bshowcheckerboard ) {
		glsub_draw_ground(3, 0.5, light_gray, dark_gray, ground_position, ground_normal);
	}
	pworld->render();
}

void DrawModelWithAllGrasps(void)
{
	if ( !pworld ) return;

	// save current hand pose and configuration
	vector<double> q;
	SE3 T;
	pworld->_phand->getJointValues(q);
	T = pworld->_phand->getEndEffectorTransform();

	// checker board
	if ( bshowcheckerboard ) {
		glsub_draw_ground(3, 0.5, light_gray, dark_gray, ground_position, ground_normal);
	}
	// multiple grasps
	pworld->_pobject->render();
	SE3 T_obj_0 = pworld->_pobject->getPose();
	for (size_t i=0; i<graspset.size(); i++) {
		pworld->_phand->placeHandWithEndEffectorTransform(T_obj_0 * graspset._grasps[i]._T);
		pworld->_phand->setJointValues(graspset._grasps[i]._preshape);
		pworld->_phand->render();
	}

	// restore hand pose and configuration
	pworld->_phand->placeHandWithEndEffectorTransform(T);
	pworld->_phand->setJointValues(q);
}

void DrawModelWithFingerPrint(void)
{
	if ( !pworld ) return;
	if ( pworld->_fingerprinthistory.size() == 0 || current_fingerprint_idx < 0 || current_fingerprint_idx >= pworld->_fingerprinthistory.size() ) {
		DrawModel();
		return;
	}

	// checker board
	if ( bshowcheckerboard ) {
		glsub_draw_ground(3, 0.5, light_gray, dark_gray, ground_position, ground_normal);
	}
	// hand
	pworld->_phand->render();
	// object
	if ( rendermode != 0 ) {
		pworld->_pobject->render();
	}

	// access to the current fingerprint data
	FingerPrint &fp = pworld->_fingerprinthistory.getFingerPrint(current_fingerprint_idx);

	// render the fingerprint
	for (size_t i=0; i<pworld->getObject()->getSurfaces().size(); i++) {
		RigidSurface *psurf = pworld->getObject()->getSurface(i);
		if ( !psurf->isEnabledCollision() || psurf->_pT == NULL ) 
			continue;

		glPushMatrix();
		glMultMatrixd(psurf->_pT->GetArray()); 

		// render faces
		glPolygonMode(GL_FRONT,GL_FILL);
		for (size_t j=0; j<psurf->getNumFaces(); j++) {

			// set face color
			if ( fp._face_hit_cnt[i][j] <= 0 ) {
				glColor3f(0.5, 0.5, 0.5);
			} else {
				if ( fp_ref_hit_cnt_for_rendering <= 0 ) { fp_ref_hit_cnt_for_rendering = 1; }
				float intensity = float(fp._face_hit_cnt[i][j]) / float(fp_ref_hit_cnt_for_rendering);
				if ( intensity > 1 ) { intensity = 1; }
				glColor3f(1, 1-intensity, 0); // inverse autumn color map (yellow --> red)
			}

			// draw face
			switch ( psurf->faces[j].elem_type ) {
			case SurfaceMesh::MT_TRIANGLE:
				glBegin(GL_TRIANGLES);
				if ( psurf->faces[j].normal_type == SurfaceMesh::NT_FACE ) {
					glNormal3dv(psurf->normals[psurf->faces[j].normal_indices[0]].GetArray());
				}
				for (int k=0; k<3; k++) {
					if ( psurf->faces[j].normal_type == SurfaceMesh::NT_VERTEX ) {
						glNormal3dv(psurf->normals[psurf->faces[j].normal_indices[k]].GetArray());
					}
					glVertex3dv(psurf->vertices[psurf->faces[j].vertex_indices[k]].GetArray());
				}
				glEnd();
				break;
			case SurfaceMesh::MT_QUAD:
				glBegin(GL_QUADS);
				if ( psurf->faces[j].normal_type == SurfaceMesh::NT_FACE ) {
					glNormal3dv(psurf->normals[psurf->faces[j].normal_indices[0]].GetArray());
				}
				for (int k=0; k<4; k++) {
					if ( psurf->faces[j].normal_type == SurfaceMesh::NT_VERTEX ) {
						glNormal3dv(psurf->normals[psurf->faces[j].normal_indices[k]].GetArray());
					}
					glVertex3dv(psurf->vertices[psurf->faces[j].vertex_indices[k]].GetArray());
				}
				glEnd();
				break;
			}
		}

		glPopMatrix();
	}

}

void UpdateRenderRefreshRateSetting(void)
{
	if ( !pworld ) return;

	// adjust the base rendering refersh rate if needed
	if ( pworld->_simuldata.size() > 0 ) {
		while ( base_render_refersh_rate > 1./time_loading_rendering ) {
			base_render_refersh_rate *= 0.5;
			cout << "base rendering refresh rate (msec) adjusted to " << base_render_refersh_rate << endl;
		}
	}

	// adjust num_skip_frames and current_render_refersh_rate using base_render_refersh_rate, replay_speed and data time step
	double data_freq;
	if ( pworld->_simuldata.get_time_step() > 1E-8 ) {
		data_freq = 1./pworld->_simuldata.get_time_step();
	} else {
		data_freq = 120;
	}
	num_skip_frames = int(replay_speed*data_freq/base_render_refersh_rate-1);
	if ( num_skip_frames < 0 ) {
		num_skip_frames = 0;
	}
	current_render_refersh_rate = base_render_refersh_rate;

	//int msec = int(1000./current_render_refersh_rate - time_loading_rendering*1000.);
	//cout << "msec = " << msec << " = " << int(1000./current_render_refersh_rate) << " - " << int(time_loading_rendering*1000.) << endl;
}

void ReplayTest()
{
	if ( !pworld ) return;

	if ( pworld->_simuldata.size() < 2 ) return;
	int maxcnt = 20;
	tic();
	for (int i=0; i<maxcnt; i++) {
		int idx = int(prand(pworld->_simuldata.size()));
		if ( idx < 0 ) idx = 0;
		double t;
		if ( !pworld->_simuldata.read_data_from_memory(t, idx) ) {
			std::cout << "failed in reading data: frame index = " << idx << std::endl;
			return;
		}
		pworld->updateKinematics();
		glutPostRedisplay();
	}
	time_loading_rendering = toc();
	time_loading_rendering /= (double)maxcnt;
	UpdateRenderRefreshRateSetting();
	ReadSimulData();
	glutPostRedisplay();
}

void OpenFileDialog(string &filepath_, bool bopen_, const char *title_, vector<string> filefilternames, vector<string> filefilterpatterns, const char *defext_ = NULL)
{
	filepath_.clear();
#ifdef _WIN32
	char filter[500];
	int cnt=0;
	for (size_t i=0; i<filefilternames.size(); i++) {
		for (size_t j=0; j<filefilternames[i].size(); j++) { filter[cnt++] = filefilternames[i][j]; } filter[cnt++] = '\0';
		for (size_t j=0; j<filefilterpatterns[i].size(); j++) { filter[cnt++] = filefilterpatterns[i][j]; } filter[cnt++] = '\0';
	}
	filter[cnt++] = '\0';
	if ( ofn.lpstrFile       ) { delete [] ofn.lpstrFile; }
    if ( ofn.lpstrInitialDir ) { delete [] ofn.lpstrInitialDir; }
    memset((void*)&ofn, 0, sizeof(ofn));
	ofn.lStructSize = sizeof(OPENFILENAME);
	if ( bopen_ ) {
		ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY | OFN_ENABLESIZING;
	} else {
		ofn.Flags = OFN_EXPLORER | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY | OFN_ENABLESIZING | OFN_OVERWRITEPROMPT;
	}
    ofn.nMaxFile     = 4096-1;
    ofn.lpstrFile    = new char[4096];
    ofn.lpstrFile[0] = 0;
	ofn.lpstrFilter  = filter;
    ofn.hwndOwner    = GetForegroundWindow();
    ofn.lpstrTitle   = title_;
	if ( !!defext_ ) {
		ofn.lpstrDefExt = defext_;
	}
    int err = GetOpenFileName(&ofn);
	if ( err == 0 ) {
        err = CommDlgExtendedError();           // extended error check
        if ( err == 0 ) return;                 // user hit 'cancel'
		cerr << "CommDlgExtendedError() code = " << err << endl;
        return;
    }
	filepath_ = string(ofn.lpstrFile);
	return;
#else
	char cmd[500];
	if ( bopen_ ) {
		sprintf(cmd, "python filechooser.py -o -t \'%s\'", title_);
	} else {
		sprintf(cmd, "python filechooser.py -s -t \'%s\'", title_);
	}
	FILE *lsofFile_p = popen(cmd, "r");
	if (!lsofFile_p) { return; }
	char buffer[4096]; char *line_p = fgets(buffer, sizeof(buffer), lsofFile_p); pclose(lsofFile_p);
	filepath_ = string(buffer);
	filepath_.erase(remove_if(filepath_.end()-1, filepath_.end(), ::isspace), filepath_.end()); // to remove the new line character in the end
#endif
}

void OpenFileDialogMultiSelection(vector<string> &filepaths_, const char *title_, vector<string> filefilternames, vector<string> filefilterpatterns, const char *defext_ = NULL)
{
	filepaths_.clear();
#ifdef _WIN32
	char filter[500];
	int cnt=0;
	for (size_t i=0; i<filefilternames.size(); i++) {
		for (size_t j=0; j<filefilternames[i].size(); j++) { filter[cnt++] = filefilternames[i][j]; } filter[cnt++] = '\0';
		for (size_t j=0; j<filefilterpatterns[i].size(); j++) { filter[cnt++] = filefilterpatterns[i][j]; } filter[cnt++] = '\0';
	}
	filter[cnt++] = '\0';
	if ( ofn2.lpstrFile       ) { delete [] ofn2.lpstrFile; }
    if ( ofn2.lpstrInitialDir ) { delete [] ofn2.lpstrInitialDir; }
    memset((void*)&ofn2, 0, sizeof(ofn2));
	ofn2.lStructSize = sizeof(OPENFILENAME);
	ofn2.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY | OFN_ENABLESIZING | OFN_ALLOWMULTISELECT;
    ofn2.nMaxFile     = 4096-1;
    ofn2.lpstrFile    = new char[10*4096];
    ofn2.lpstrFile[0] = 0;
	ofn2.lpstrFilter  = filter;
    ofn2.hwndOwner    = GetForegroundWindow();
    ofn2.lpstrTitle   = title_;
	if ( !!defext_ ) {
		ofn2.lpstrDefExt = defext_;
	}
    int err = GetOpenFileName(&ofn2);
	if ( err == 0 ) {
        err = CommDlgExtendedError();           // extended error check
        if ( err == 0 ) return;                 // user hit 'cancel'
		cerr << "CommDlgExtendedError() code = " << err << endl;
        return;
    }

	bool bMultipleFileSelected = (ofn2.lpstrFile[ofn2.nFileOffset - 1] == '\0');
	if (bMultipleFileSelected) {
		bool bDirectoryIsRoot = (ofn2.lpstrFile[strlen(ofn2.lpstrFile) - 1] == '\\');
		for(CHAR *szTemp = ofn2.lpstrFile + ofn2.nFileOffset; *szTemp; szTemp += (strlen(szTemp) + 1)) {
			size_t dwLen = strlen(ofn2.lpstrFile) + strlen(szTemp) + 2;
			CHAR * szFile = new CHAR[dwLen];
			strcpy_s(szFile, dwLen, ofn2.lpstrFile);
			if (!bDirectoryIsRoot) {
				strcat_s(szFile, dwLen, "\\"); 
			}
			strcat_s(szFile, dwLen, szTemp);   
			filepaths_.push_back(string(szFile));
			//cout << string(szFile) << endl;
			delete szFile;
		}
	} else {
		filepaths_.push_back(string(ofn2.lpstrFile));
		//cout << string(ofn2.lpstrFile) << endl;
	} 
	return;
#else
	char cmd[500];
	sprintf(cmd, "python filechooser.py -o -m -t \'%s\'", title_);
	FILE *lsofFile_p = popen(cmd, "r");
	if (!lsofFile_p) { return; }
	char buffer[4096]; char *line_p = fgets(buffer, sizeof(buffer), lsofFile_p); pclose(lsofFile_p);
	char tmp[1000]; int cnt=0;
	for (int i=0; i<4096; i++) {
		if ( buffer[i] == '\n' ) {
			break;
		}
		if ( buffer[i] == ':' ) { 
			filepaths_.push_back(string(tmp));
			cnt = 0; 
			continue; 
		}
		tmp[cnt++] = buffer[i];
	}
#endif
}


void OpenFolderDialog(string &folderpath, const char *title)
{
	folderpath.clear();
	char cCurrentPath[4096];
	GetCurrentDir(cCurrentPath, sizeof(cCurrentPath));
#ifdef _WIN32
	TCHAR szDir[4096];
	bInfo.hwndOwner = NULL;
	bInfo.pidlRoot = NULL; 
	bInfo.pszDisplayName = szDir; // Address of a buffer to receive the display name of the folder selected by the user
	bInfo.lpszTitle = title;
	bInfo.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;//BIF_USENEWUI;
	bInfo.lpfn = BrowseCallbackProc;//NULL;
	bInfo.lParam = (LPARAM)cCurrentPath;//NULL;
	bInfo.iImage = -1;
	LPITEMIDLIST lpItem = SHBrowseForFolder(&bInfo);
	if( lpItem != NULL )
	{
	  SHGetPathFromIDList(lpItem, szDir);
	  folderpath = string(szDir);
	}
#else
	char cmd[500];
	sprintf(cmd, "python filechooser.py -d -t \'%s\'", title);
	FILE *lsofFile_p = popen(cmd, "r");
	if (!lsofFile_p) { return; }
	char buffer[4096]; char *line_p = fgets(buffer, sizeof(buffer), lsofFile_p); pclose(lsofFile_p);
	folderpath = string(buffer);
	folderpath.erase(remove_if(folderpath.end()-1, folderpath.end(), ::isspace), folderpath.end()); // to remove the new line character in the end
#endif
}

void CreateFolder(const char *folderpath)
{
#ifdef _WIN32
	//::CreateDirectory(folderpath, NULL); // this doesn't support recursive creation
	char cmd[500];
	sprintf(cmd, "mkdir \"%s\"", folderpath);
	system(cmd);
#else
	char cmd[500];
	sprintf(cmd, "mkdir -p \'%s\'", folderpath);
	system(cmd);
#endif
}

string ExtractDirectoryPath(const string &filepath)
{
	int idx = filepath.size()-1;
	for (; idx>=0; idx--) {
		if ( filepath[idx] == '\\' || filepath[idx] == '/' )
			break;
	}
	string dirpath = filepath;
	dirpath.erase(dirpath.begin()+idx+1, dirpath.end());
	return dirpath;
}

void LoadModel(const char *filepath)
{
	if ( !pworld ) delete pworld;
	pworld = new World;
	if ( !pworld->loadFromXML(filepath) ) {
		cerr << "failed in loading file: " << filepath << endl;
		delete pworld;
		pworld = NULL;
	}

	mvmatrix.SetRotation(SO3(-0.798338, -0.408694, 0.442296, 0.600147, -0.479198, 0.640463, -0.0498058, 0.776748, 0.627839));
	sdepth = -pworld->getBoundingSphereRadius() / tan(0.25f*fovy/180.f*3.14f);
	mscale = (float)pworld->getBoundingSphereRadius();
	max_counter_save = int(1./(double(pworld->_datasavefreq)*pworld->getStepSize()));
	bdatasavecontact = pworld->_datasavecontact;
	ReplayTest();
	//UpdateRenderRefreshRateSetting();
}

void LoadSimulDataFromFile(const char *filepath)
{
	if ( !pworld ) return;

	bool loadsuccess;
	pworld->_simuldata.set_filepath_for_reading(filepath);
	loadsuccess = pworld->_simuldata.load_data_on_memory_from_file();
	if ( !loadsuccess ) {
		// if failed in loading, try again after changing the contact saving option
		pworld->_datasavecontact = !pworld->_datasavecontact;
		pworld->_setSimulationSave();
		loadsuccess = pworld->_simuldata.load_data_on_memory_from_file();
		// if still failed,
		if ( !loadsuccess ) {
			cout << "failed in loading a simulation data file: " << filepath << endl;
			// if the contact saving option has been changed, restore it quietly
			if ( pworld->_datasavecontact != bdatasavecontact ) {
				pworld->_datasavecontact = bdatasavecontact;
				pworld->_setSimulationSave();
			}
			return;
		} 
	}
	cout << "simulation data loaded from " << filepath << endl;
	// if the contact saving option has been changed, warn it
	if ( pworld->_datasavecontact != bdatasavecontact ) {
		cout << "warning:: contact saving option has been changed to load the data! (datasavecontact = " << (int)pworld->_datasavecontact << ")" << endl;
		cout << "          (you can restore the setting by pressing F2)" << endl;
	}

	current_frame_idx = 0;
	ReadSimulData();
	ReplayTest();
}

void LoadSimulDataFromFiles(vector<string> filepaths)
{
	if ( !pworld ) return;

	pworld->_simuldata.clear_data_on_memory();

	for (size_t i=0; i<filepaths.size(); i++) {
		bool loadsuccess;
		pworld->_simuldata.set_filepath_for_reading(filepaths[i].c_str());
		loadsuccess = pworld->_simuldata.load_data_on_memory_from_file_app();
		if ( !loadsuccess ) {
			// if failed in loading, try again after changing the contact saving option
			pworld->_datasavecontact = !pworld->_datasavecontact;
			pworld->_setSimulationSave();
			loadsuccess = pworld->_simuldata.load_data_on_memory_from_file_app();
			// if still failed,
			if ( !loadsuccess ) {
				cout << "failed in loading a simulation data file: " << filepaths[i] << endl;
			}
		}
		cout << "simulation data loaded from " << filepaths[i] << endl;
	}
	// if the contact saving option has been changed, warn it
	if ( pworld->_datasavecontact != bdatasavecontact ) {
		cout << "warning:: contact saving option has been changed to load the data! (datasavecontact = " << (int)pworld->_datasavecontact << ")" << endl;
		cout << "          (you can restore the setting by pressing F2)" << endl;
	}

	current_frame_idx = 0;
	ReadSimulData();
	ReplayTest();
}

void ReadSimulData(void)
{
	if ( !pworld ) return;

	if ( pworld->_simuldata.size() == 0 ) {
		return;
	}
	if ( !pworld->_simuldata.read_data_from_memory(current_time, current_frame_idx) ) {
		std::cout << "failed in reading data: frame index = " << current_frame_idx << std::endl;
		return;
	}
	pworld->updateKinematics();

	if ( pworld->_simuldata.size() > 0 ) {
		sliderX = int( (float)sliderXi + (float)(sliderXf-sliderXi) * (float)current_frame_idx / (float)(pworld->_simuldata.size()-1) );
	} else {
		sliderX = sliderXi;
	}
}

void StopSimulation(void)
{
	if ( !pworld ) return;

	glutIdleFunc(NULL);
	gReal elapsed_time = toc();
	end_simul_time = current_time;
	std::cout << "computation time for " << end_simul_time - start_simul_time << "sec simulation = " << elapsed_time << " sec" << std::endl;
	bsimul = false;
	current_frame_idx = pworld->_simuldata.size()-1;
	ReadSimulData();

	// reset current surface contact info not to confuse rendering in replay
	if ( !!pworld && !pworld->_datasavecontact ) {
		for (size_t i=0; i<pworld->_colchecker.getSurfaces().size(); i++) {
			pworld->_colchecker.getSurfaces()[i]->resetContactInfo();
		}
	}

	ReplayTest();
	cout << "simulation result summary: " << endl;
	cout << " number of contact links after grasping = " << pworld->_num_contact_links_after_liftup << endl;
	cout << " final pose deviation = ";
	cout << pworld->_deviation_pos_final << " (m), " << pworld->_deviation_ori_final * 180. / 3.14159 << " (deg)" << endl;
	cout << " max pose deviation during grasping = ";
	cout << pworld->_deviation_pos_max_grasping << " (m), " << pworld->_deviation_ori_max_grasping * 180. / 3.14159 << " (deg)" << endl;
	if ( pworld->_apply_external_force && ( pworld->_gq_external_forces.size() != 0 || pworld->_gq_external_moments.size() != 0 ) ) {
		cout << " response to external forces (max deviation (m), resistant time (sec)):" << endl;
		for (size_t i=0; i<pworld->_gq_external_forces.size(); i++) {
			cout << "  " << setw(12);
			if ( pworld->_deviation_pos_max_externalforce[i] >= 0 ) {
				cout << pworld->_deviation_pos_max_externalforce[i];
			} else {
				cout << "none";
			}
			cout << ", " << setw(12);
			if ( pworld->_time_resistant_externalforce[i] >= 0 ) {
				cout << pworld->_time_resistant_externalforce[i];
			} else {
				cout << "none";
			}
			cout << endl;
		}
		cout << " response to external moments (max deviation (deg), resistant time (sec)):" << endl;
		for (size_t i=0; i<pworld->_gq_external_moments.size(); i++) {
			cout << "  " << setw(12);
			if ( pworld->_deviation_ori_max_externalmoment[i] >= 0 ) {
				cout << pworld->_deviation_ori_max_externalmoment[i] * 180. / 3.14159;
			} else {
				cout << "none";
			}
			cout << ", " << setw(12);
			if ( pworld->_time_resistant_externalmoment[i] >= 0 ) {
				cout << pworld->_time_resistant_externalmoment[i];
			} else {
				cout << "none";
			}
			cout << endl;
		}
	}
	cout << " min distance (force-closure) after grasping = " << pworld->_min_dist_after_liftup << endl;
	cout << " grasp quality scores [sA, sBp, sBr, sCf, sCt, sDf, sDt, sE]:" << endl;
	vector<double> qscores; pworld->scoreGraspQuality(qscores);
	for (size_t j=0; j<qscores.size(); j++) {
		cout << setw(8) << qscores[j] << " ";
	}
	cout << endl << endl;
}

void StartSimulation(void)
{
	if ( !pworld ) return;

	pworld->_simuldata.truncate_data_on_memory(current_frame_idx);
	pworld->_simuldata.write_data_into_memory(current_time);
	counter_save = 0;
	bsimul = true;
	start_simul_time = current_time;
	tic();
	glutIdleFunc(StepSimulation);
}

void StepSimulation(void)
{
	if ( !pworld ) return;

	int cnt=0;
	while ( cnt++ < max_counter_save ) {
		bool bsuccess = true;
		if ( !pworld->stepSimulation(current_time) ) {
			cout << "simulation stopped! (t = " << current_time << " sec)" << endl;
			bsuccess = false;
		}
		current_time += pworld->getStepSize();
		if ( !bsuccess || pworld->isSimulationDone() || current_time > max_simul_time ) {
			if ( current_time > max_simul_time ) {
				cout << "reached maximum simulation time (" << max_simul_time << "sec)!" << endl;
			}
			pworld->_simuldata.write_data_into_memory(current_time);
			StopSimulation();
			glutPostRedisplay();
			return;
		}
	}
	pworld->_simuldata.write_data_into_memory(current_time);
}

void ApplyGrasp(int idx)
{
	if ( !pworld ) return;

	if ( idx < 0 || idx >= graspset.size() ) {
		cerr << "invalid grasp index" << endl;
		return;
	}
	SE3 T_ee = pworld->getObject()->getPose() * graspset._grasps[idx]._T;
	pworld->getHand()->placeHandWithEndEffectorTransform(T_ee);
	if ( !pworld->getHand()->setJointValues(graspset._grasps[idx]._preshape) ) { 
		cerr << "failed in setting joint values with preshape!" << endl;
		return;
	}
}

void MeasureGraspQuality(string folderpath = string(), bool bsavesimul = true, bool bappend = false)
{
	if ( !pworld ) return;

	// select a folder for saving grasp quality data
	if ( folderpath.size() == 0 ) {
		OpenFolderDialog(folderpath, "Select a folder for saving grasp quality data");
	}
	if ( folderpath.size() == 0 ) {
		std::cout << "A folder must be selected to proceed!" << endl;
		return;
	}

	if ( !bsavesimul ) {
		cout << endl;
		cout << "warning:: grasp simulation data will not be saved as requested!" << endl;
		cout << endl;
	}

	// change options
	int datasavefreq_prev = pworld->_datasavefreq;
	bool datasavecontact_prev = pworld->_datasavecontact;
	if ( pworld->_datasavefreq > 60 ) { pworld->_datasavefreq = 60; }
	pworld->_datasavecontact = false;
	pworld->_setSimulationSave();
	pworld->getHand()->showMotorMessage(false);

	// temp grasp set
	GraspSet graspsetA, graspsetBp, graspsetBr, graspsetAB, graspsetE;
	graspsetA = graspsetBp = graspsetBr = graspsetAB = graspsetE = graspset;

	// measure grasps
	int idx_start = 0;
	std::cout << "measuring grasp quality started..." << endl;
	string ucfile = folderpath; ucfile += "/objpose_uncertainty.m";
	pworld->_obj_pose_uncertainty.save(ucfile.c_str());
	string logfile = folderpath; logfile += "/simlog.m";
	string proglogfile = folderpath; proglogfile += "/proglog.m"; // progress log
	ofstream fout, foutprog;
	if ( bappend && fexists(logfile.c_str()) && fexists(proglogfile.c_str()) ) {
		// check progress of the previous stage
		ifstream fin(proglogfile.c_str());
		int idx_last = -1;
		while ( !fin.eof() ) { fin >> idx_last; }
		fin.close();
		if ( idx_last < 0 || idx_last >= graspset.size() ) {
			cout << "app mode: invalid progress information!" << endl;
			cout << "check data folder!" << endl;
			return;
		} else {
			fout.open(logfile.c_str(), ios::app);
			cout << "app mode: existing main log file opend for appending" << endl;
			fout << "% ---------- data appended below ---------------------" << endl;
			foutprog.open(proglogfile.c_str(), ios::app);
			idx_start = idx_last + 1; // start from the next grasp
		}
	} else {
		fout.open(logfile.c_str());
		fout << "% grasp_index, grasp_quality" << endl << endl;
		fout << "simlogdata = [" << endl;
		foutprog.open(proglogfile.c_str());
	}
	tic();
	bsimul = true;
	for (size_t i=idx_start; i<graspset.size(); i++) {
		std::cout << "----- grasp " << i << " -----" << endl;
		// create a sub-folder where the simulation files to be saved
		stringstream format; format << "%0" << ceil(log10((float)graspset.size())) << "d";
		char str[10]; sprintf(str, format.str().c_str(), i);
		stringstream ssfolderpath; ssfolderpath << folderpath << "/grasp_" << str; // sub-folder name
		if ( bsavesimul ) {
			CreateFolder(ssfolderpath.str().c_str());
		}
		// measure the grasp quality 
		vector<double> quality_scores;
		// call the simulation-based grasp quality measuring algorithm
		pworld->measureGraspQuality(quality_scores, graspset._grasps[i], bsavesimul, ssfolderpath.str().c_str());
		// print the quality
		fout << setw(5) << i << " ";
		for (size_t j=0; j<quality_scores.size(); j++) {
			fout << setw(12) << quality_scores[j] << " ";
		}
		fout << endl;
		// save averaged quality score for sorting later
		if ( quality_scores.size() > 3 ) {
			graspsetA._grasps[i]._score = quality_scores[0];
			graspsetBp._grasps[i]._score = quality_scores[1];
			graspsetBr._grasps[i]._score = quality_scores[2];
			graspsetAB._grasps[i]._score = 0.5 * quality_scores[0] + 0.25 * (quality_scores[1] + quality_scores[2]);
			graspsetE._grasps[i]._score = quality_scores[7];
			graspset._grasps[i]._score = (quality_scores[0] + quality_scores[1] + quality_scores[2]) / 3.; 
		}
		// record the current progress
		foutprog << i << endl;
	}
	double t = toc();
	std::cout << "elapsed time for measuring the quality of " << graspset.size() << " grasps = " << t << " sec" << endl;
	fout << "];" << endl;
	fout << "elapsed_time = " << t << ";" << endl;
	fout.close();
	foutprog.close();
	bsimul = false;

	// sort grasps
	std::cout << "sorting grasps based on the quality score..." << endl;
	graspset.sort();
	graspsetA.sort();
	graspsetBp.sort();
	graspsetBr.sort();
	graspsetAB.sort();
	graspsetE.sort();
	std::cout << "done" << endl;

	// save grasps
	string filepath;
	filepath = folderpath + "/graspset_sorted.gst";
	graspset.save(filepath.c_str());
	std::cout << "sorted grasps saved to " << filepath << endl;
	filepath = folderpath + "/graspset_sorted_A.gst";
	graspsetA.save(filepath.c_str());
	std::cout << "sorted grasps saved to " << filepath << endl;
	filepath = folderpath + "/graspset_sorted_Bp.gst";
	graspsetBp.save(filepath.c_str());
	std::cout << "sorted grasps saved to " << filepath << endl;
	filepath = folderpath + "/graspset_sorted_Br.gst";
	graspsetBr.save(filepath.c_str());
	std::cout << "sorted grasps saved to " << filepath << endl;
	filepath = folderpath + "/graspset_sorted_AB.gst";
	graspsetAB.save(filepath.c_str());
	std::cout << "sorted grasps saved to " << filepath << endl;
	filepath = folderpath + "/graspset_sorted_E.gst";
	graspsetE.save(filepath.c_str());
	std::cout << "sorted grasps saved to " << filepath << endl;

	// restore options
	pworld->_datasavecontact = datasavecontact_prev;
	pworld->_datasavefreq = datasavefreq_prev;
	pworld->_setSimulationSave();
	pworld->getHand()->showMotorMessage(bshowmotormsg);
}

void MeasureGraspQualityWithoutUncertainty(string folderpath = string())
{
	if ( !pworld ) return;

	// select a folder for saving simulation data
	if ( folderpath.size() == 0 ) {
		OpenFolderDialog(folderpath, "Select a folder for saving simulation data");
	}
	if ( folderpath.size() == 0 ) {
		std::cout << "A folder must be selected to proceed!" << endl;
		return;
	}

	// measure grasps without uncertainty
	std::cout << "measuring grasp quality (without uncertainty) started..." << endl;
	CreateFolder(folderpath.c_str());
	tic();
	bsimul = true;
	pworld->measureGraspQualityWithoutUncertainty(graspset, folderpath.c_str());
	double t = toc();
	std::cout << "elapsed time = " << t << " sec" << endl;
	bsimul = false;
}

void StopReplay(void)
{
	if ( !pworld ) return;
	breplay = false;
}

void StartReplay(void)
{
	if ( !pworld ) return;
	breplay = true;
}

void ResetTime(void) 
{
	if ( !pworld ) return;
	current_frame_idx = 0;
	ReadSimulData();
	glutPostRedisplay();
}

void SaveContactForceData(string filepath)
{
	int idx_prev = current_frame_idx;
	ofstream fout(filepath.c_str());

	bool bmatlab = false;
	if ( filepath.size() > 2 && filepath[filepath.size()-1] == 'm' && filepath[filepath.size()-2] == '.' ) {
		bmatlab = true;
	}
	if ( bmatlab ) {
		string filename = filepath; 
		filename.erase(filepath.size()-2, 2);
		fout << "function data = " << filename << "(fidx)" << endl;
		fout << "if nargin < 1" << endl;
		fout << "  fidx = 1;" << endl;
		fout << "end" << endl;
	} else {
		fout << "fidx = 1;" << endl;
	}

	fout << "data = [" << endl;
	for (int i=0; i<pworld->_simuldata.size(); i++) {
		current_frame_idx = i;
		ReadSimulData();
		Vec3 f;
		fout << current_time << "   ";
		RigidBody *pbody; 
		pbody = pworld->_phand->getBody("finger0-2");
		f = pbody->getOrientationGlobal() * pbody->Fe.GetF(); // non-filtered contact force
		fout << f[0] << " " << f[1] << " " << f[2] << "   ";
		pbody = pworld->_phand->getBody("finger1-2");
		f = pbody->getOrientationGlobal() * pbody->Fe.GetF();
		fout << f[0] << " " << f[1] << " " << f[2] << "   ";
		pbody = pworld->_phand->getBody("finger2-2");
		f = pbody->getOrientationGlobal() * pbody->Fe.GetF();
		fout << f[0] << " " << f[1] << " " << f[2] << "   ";
		for (size_t i=0; i<pworld->_phand->_motors.size(); i++) {
			for (size_t j=0; j<pworld->_phand->_motors[i]._data_debugging.size(); j++) {
				fout << pworld->_phand->_motors[i]._data_debugging[j] << " ";
			}
			fout << "  ";
		}
		fout << endl;
	}
	fout << "];" << endl;
	fout << endl;
	fout << "% plot data" << endl;
	fout << "% fidx must be defined above" << endl;
	fout << "t = data(:,1);" << endl;
	fout << "f1 = data(:,2:4);" << endl;
	fout << "f2 = data(:,5:7);" << endl;
	fout << "f3 = data(:,8:10);" << endl;
	int idx = 11;
	for (size_t i=0; i<pworld->_phand->_motors.size(); i++) {
		int size = pworld->_phand->_motors[i]._data_debugging.size();
		fout << "md" << i+1 << " = data(:," << idx << ":" << idx+size-1 << ");" << endl;
		idx += size;
	}
	fout << "figure(fidx); " << endl;
	fout << "subplot(3,1,1); plot(t,f1); grid on; legend('x','y','z'); title('force (finger0-2)');" << endl;
	fout << "subplot(3,1,2); plot(t,f2); grid on; legend('x','y','z'); title('force (finger1-2)');" << endl;
	fout << "subplot(3,1,3); plot(t,f3); grid on; legend('x','y','z'); title('force (finger2-2)');" << endl;
	fout << "figure(fidx+1);" << endl;
	fout << "subplot(2,1,1); plot(t,[md1(:,2),md2(:,2),md3(:,2)]); grid on; legend('tau1','tau2','tau3'); title('motor torque');" << endl;
	fout << "subplot(2,1,2); plot(t,[md1(:,3),md2(:,3),md3(:,3)]); grid on; legend('dqm1','dqm2','dqm3'); title('motor speed');" << endl;
	fout.close();
	current_frame_idx = idx_prev;
	ReadSimulData();
	glutPostRedisplay();
	cout << "force data saved to " << filepath << endl;
}

#include "cdfunc.h"
void TestFunc()
{
	if ( !pworld ) return;

	GRASPANALYSIS ga = pworld->AnalyzeContacts();
	cout << "mindist = " << ga.mindist << endl;
	cout << "volume = " << ga.volume << endl;

	//// error tracking
	//if ( graspset.size() == 0 || current_grasp_idx < 0 ) {
	//	cerr << "error:: no grasp given!" << endl;
	//	return;
	//}
	//string filepath;
	//vector<string> filefilternames, filefilterpatterns;
	//filefilternames.push_back("m Files (*.m)");
	//filefilterpatterns.push_back("*.m");
	//OpenFileDialog(filepath, false, "Save Data As", filefilternames, filefilterpatterns, "m");
	//if ( filepath.size() > 0 ) {
	//	cout << "start error tracking..." << endl;
	//	pworld->computeGraspQualityErrorBar(20, 2000, graspset._grasps[current_grasp_idx], filepath.c_str());
	//	cout << "error tracking data saved to " << filepath << endl;
	//}

	//cout << "mvmatrix = " << mvmatrix << endl;

	//SaveGraspDistributionForMeasuringGraspQuality();

	//// set radius of the collision vertices of the hand
	//for (list<GBody*>::iterator iter_pbody = pworld->getHand()->pBodies.begin(); iter_pbody != pworld->getHand()->pBodies.end(); iter_pbody++) {
	//	for (size_t i=0; i<((RigidBody*)(*iter_pbody))->getSurfaces().size(); i++) {
	//		((RigidBody*)(*iter_pbody))->getSurface(i)->vertex_radius = 0.001;
	//	}
	//}

	//// compuate vertex normals of the contact points of the hand
	//ofstream fout("handpointinfo.txt");
	//vector< vector<Vec3> > P(pworld->getHand()->_pbodies_new.size()), N(pworld->getHand()->_pbodies_new.size());
	//vector< vector<double> > D(pworld->getHand()->_pbodies_new.size());
	//for (size_t i=0; i<pworld->getHand()->_pbodies_new.size(); i++) {
	//	fout << "link: " << pworld->getHand()->_pbodies_new[i]->getName() << endl;
	//	RigidSurface *psurf0 = pworld->getHand()->_pbodies_new[i]->pSurfs[0];
	//	RigidSurface *psurf1 = pworld->getHand()->_pbodies_new[i]->pSurfs[1];
	//	SE3 T = *(psurf0->_pT);
	//	SE3 invT = Inv(T);
	//	P[i].resize(psurf1->getNumVertices()); 
	//	N[i].resize(psurf1->getNumVertices());
	//	D[i].resize(psurf1->getNumVertices(), 100);
	//	for (size_t j=0; j<psurf1->getNumVertices(); j++) {
	//		Vec3 x = T * psurf1->vertices[j];
	//		P[i][j] = psurf1->vertices[j];
	//		for (size_t k=0; k<psurf0->getNumFaces(); k++) {
	//			Vec3 t0 = T * psurf0->vertices[psurf0->faces[k].vertex_indices[0]];
	//			Vec3 t1 = T * psurf0->vertices[psurf0->faces[k].vertex_indices[1]];
	//			Vec3 t2 = T * psurf0->vertices[psurf0->faces[k].vertex_indices[2]];

	//			Vec3 n; double d;
	//			if ( dcPointTriangle(d, n, x, t0, t1, t2) && fabs(d) < fabs(D[i][j]) ) {
	//				N[i][j] = invT * n;
	//				D[i][j] = d;
	//			}
	//		}
	//		if ( fabs(D[i][j]) > 1E-4 ) {
	//			cout << "warning:: link: " << pworld->getHand()->_pbodies_new[i]->getName() << ", vertex " << j << ": d = " << D[i][j] << endl;
	//			fout << "warning:: link: " << pworld->getHand()->_pbodies_new[i]->getName() << ", vertex " << j << ": d = " << D[i][j] << endl;
	//		}
	//	}

	//	fout << "<vertices>";
	//	for (size_t j=0; j<psurf1->getNumVertices(); j++) {
	//		fout << P[i][j][0] << " " << P[i][j][1] << " " << P[i][j][2] << " ";
	//	}
	//	fout << "</vertices>" << endl;
	//	fout << "<vertexnormals>";
	//	for (size_t j=0; j<psurf1->getNumVertices(); j++) {
	//		fout << N[i][j][0] << " " << N[i][j][1] << " " << N[i][j][2] << " ";
	//	}
	//	fout << "</vertexnormals>" << endl;
	//	fout << endl;
	//}
	//fout.close();
}

void Timer(int extra) 
{
	if ( !pworld ) {
		glutTimerFunc(int(1000./30.), Timer, 0);
		return;
	}

	if ( breplay ) {
		current_frame_idx += 1 + num_skip_frames;
		if ( current_frame_idx >= pworld->_simuldata.size() ) {
			breplay = false;
			current_frame_idx = pworld->_simuldata.size()-1;
		}
		ReadSimulData();
		glutPostRedisplay();
		int msec = int(1000./current_render_refersh_rate - time_loading_rendering*1000.);
		if ( msec < 0 ) msec = 0;
		glutTimerFunc(msec, Timer, 0);
		return;
	}
	if ( bsimul ) {
		glutPostRedisplay();
		glutTimerFunc(int(1000./10.), Timer, 0);
		return;
	}
	if ( breplay && bsimul ) {
		breplay = bsimul = false;
	}
	glutTimerFunc(int(1000./30.), Timer, 0);
}

void ReshapeCallback(int width, int height) 
{
	winsizeX = width;
	winsizeY = height;
	aspect = (float)winsizeX/(float)winsizeY;
	glViewport(0, 0, winsizeX, winsizeY);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glutPostRedisplay();
}
 
void DisplayCallback(void) 
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fovy, aspect, zNear, zFar);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMultMatrixd(mvmatrix.GetArray());
	if ( bshowallgrasps ) {
		DrawModelWithAllGrasps();
	} 
	else if ( bshowfingerprint ) {
		DrawModelWithFingerPrint();
	}
	else {
		DrawModel();
	}
	DrawTextInfo();
	glutSwapBuffers();
	glClearColor(backgroundcolor[0],backgroundcolor[1],backgroundcolor[2],backgroundcolor[3]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
}

void SetAlphaRendering(float alpha_hand, float alpha_object) {
	for (size_t i=0; i<pworld->_phand->_pbodies_new.size(); i++) {
		for (size_t j=0; j<pworld->_phand->_pbodies_new[i]->pSurfs.size(); j++) {
			pworld->_phand->_pbodies_new[i]->pSurfs[j]->alpha = alpha_hand;
		}
	}
	for (size_t i=0; i<pworld->_pobject->_pSurfs.size(); i++) {
		pworld->_pobject->_pSurfs[i]->alpha = alpha_object;
	}
}

void KeyboardCallback(unsigned char ch, int x, int y) 
{
	if (glutGetModifiers() == GLUT_ACTIVE_CTRL) { 
		bctrlpressed = true;
		ch = (char)((int)ch+96);
	}
	if ( glutGetModifiers() == GLUT_ACTIVE_SHIFT) {
		bshiftpressed = true;
		ch = (char)((int)ch+32);
	}
	if ( glutGetModifiers() == GLUT_ACTIVE_ALT) {
		baltpressed = true;
	}

	if ( !pworld ) {
		if ( !(ch == 'o' && bshiftpressed) && ch != KEY_ESC && ch != 'u' ) {
			cout << "Simulation world not defined yet! (load model by pressing shift+o)" << endl;
			return;
		}
	}

	string filepath, strtmp, folderpath;
	vector<string> filefilternames, filefilterpatterns, filepaths;
	vector<int> dup;
	
	switch (ch) {
	case KEY_ESC:// exit program
		cout << "exit program" << endl;
		exit(0);
		break;
	case KEY_ENTER:  // start/pause simulation
		if ( bsimul ) {
			StopSimulation();
		} else {
			if ( breplay ) {
				StopReplay();
			} 
			StartSimulation();
		}
		break;
	case KEY_SPACE: // play/pause replay
		if ( breplay ) {
			StopReplay();
		} else {
			if ( bsimul ) {
				StopSimulation();
			}
			if ( current_frame_idx == pworld->_simuldata.size()-1 ) {
				ResetTime();
			}
			StartReplay();
		}
		break;
	case 'b': // show/hide bounding boxes
		bshowboundingbox = !bshowboundingbox;
		pworld->enableRenderingBoundingBoxes(bshowboundingbox);
		break;
	case 'c': // show/hide contact forces, contact surface patch, and grasp contact center after grasping and before lifting up
		if ( bctrlpressed ) {
			bshowcontactsurfacepatch = !bshowcontactsurfacepatch;
			pworld->enableRenderingCollidingSurfacePatches(bshowcontactsurfacepatch);
		} 
		else if ( bshiftpressed ) {
			pworld->_b_show_gcc_obj = !pworld->_b_show_gcc_obj;
		}
		else {
			bshowcontactforces = !bshowcontactforces;
			pworld->enableRenderingContactForces(bshowcontactforces);
		}
		break;
	case 'd': // toggle draw type (original/wire/collision/all)
		rendermode++;
		if ( rendermode > 3 ) rendermode = 0;
		pworld->setRenderingMode(rendermode);
		break;
	case 'e': // error tracking analysis for the current grasp
		if ( bctrlpressed ) {
			if ( graspset.size() == 0 || current_grasp_idx < 0 ) {
				cerr << "error:: no grasp given!" << endl;
				return;
			}
			filepath.clear();
			filefilternames.push_back("m Files (*.m)");
			filefilterpatterns.push_back("*.m");
			OpenFileDialog(filepath, false, "Save Data As", filefilternames, filefilterpatterns, "m");
			if ( filepath.size() > 0 ) {
				cout << "start error tracking... (20 seeds, up to 2000 samples)" << endl;
				cout << "output file = " << filepath << endl;
				pworld->computeGraspQualityErrorBar(20, 2000, graspset._grasps[current_grasp_idx], filepath.c_str());
				cout << "error tracking data saved to " << filepath << endl;
			}
		}
		break;
	case 'f': 
		if ( bctrlpressed ) { // calculate and save fingerprint history of the current grasp set
			if ( graspset.size() == 0 ) {
				cout << "empty grasp set!" << endl;
				return;
			}
			// set file to save fingerprint history
			filepath.clear();
			filefilternames.push_back("Fingerprint History File (*.fph)");
			filefilterpatterns.push_back("*.fph");
			filefilternames.push_back("All Files (*.*)");
			filefilterpatterns.push_back("*.*");
			OpenFileDialog(filepath, false, "Save Fingerprint History As", filefilternames, filefilterpatterns, "fph");
			if ( filepath.size() > 0 ) {
				GraspSet gs_finalconfig;
				if ( !pworld->calcFingerPrintHistory(graspset, gs_finalconfig) ) {
					cerr << "error:: failed in calculating fingerprint history" << endl;
					return;
				}
				pworld->_fingerprinthistory.save(filepath.c_str());
				cout << "fingerprint history saved to " << filepath << endl;
				string gsfilepath = filepath; gsfilepath += ".gst";
				gs_finalconfig.save(gsfilepath.c_str());
				cout << "final grasp configurations saved to " << gsfilepath << endl;
				bshowfingerprint = true;
				current_fingerprint_idx = -1;
				fp_ref_hit_cnt_for_rendering = int(fp_ref_hit_cnt_ratio_init*float(pworld->_fingerprinthistory.size()));
				glutPostRedisplay();
			}
		} 
		else { // save force data
			SaveContactForceData("fdata.m");
		}
		break;
	case 'g': // show/hide ground
		bshowcheckerboard = !bshowcheckerboard;
		pworld->_ground.bRendering = !pworld->_ground.bRendering;
		break;
	case 'h': // show/hide hand or object
		if ( pworld->_phand->_bRendering && pworld->_pobject->_bRendering) {
			pworld->_phand->_bRendering = false;
		} else if ( !pworld->_phand->_bRendering && pworld->_pobject->_bRendering ) {
			pworld->_phand->_bRendering = true;
			pworld->_pobject->_bRendering = false;
		} else {
			pworld->_phand->_bRendering = true;
			pworld->_pobject->_bRendering = true;
		}
		break;
	case 'i': // print model information
		if ( bctrlpressed ) {
			bshowmotormsg = !bshowmotormsg;
			pworld->getHand()->showMotorMessage(bshowmotormsg);
			if ( bshowmotormsg ) {
				cout << "motor messages will be shown during simulation" << endl;
			} else {
				cout << "motor messages will NOT be shown during simulation" << endl;
			}
		} else {
			ShowModelInfo();
		}
		break;
	case 'k': // remove invalide grasps in the grasp set 
		if ( bctrlpressed ) {
			GraspSet graspset_new;
			for (size_t i=0; i<graspset.size(); i++) {
				ApplyGrasp(i);
				if ( !pworld->isHandCollidingWithGround_BoundingBoxCheckOnly() ) {
					graspset_new._grasps.push_back(graspset._grasps[i]);
				}					
			}
			cout << graspset.size()-graspset_new._grasps.size() << " grasps removed!" << endl;
			graspset = graspset_new;
			if ( graspset.size() > 0 ) {
				current_grasp_idx = 0;
				ApplyGrasp(0);
			}
		} else {
			if ( current_grasp_idx >= 0 && current_grasp_idx < graspset.size() ) {
				graspset._grasps.erase(graspset._grasps.begin()+current_grasp_idx);
				cout << "grasp " << current_grasp_idx << " removed!" << endl;
				current_grasp_idx--;
				if ( current_grasp_idx < 0 ) 
					current_grasp_idx = 0;
				ApplyGrasp(current_grasp_idx);
			}
		}
		break;
	case 'm': 
		if ( bctrlpressed ) { // Ctrl + m
			// measure grasp quality of the grasps with considering uncertainty
			MeasureGraspQuality(string(), true, false);
		}
		else if ( bshiftpressed ) { // Shift + m
			// measure grasp quality of the grasps with considering uncertainty
			MeasureGraspQuality(string(), false, false); // no simulation data saving
		}
		else if ( baltpressed ) {
			// measure grasp quality of the grasps without considering uncertainty
			MeasureGraspQualityWithoutUncertainty();
		}
		break;
	case 'n':
		//for (size_t i=0; i<pworld->_colchecker.getSurfaces().size(); i++) {
		//	pworld->_colchecker.getSurfaces()[i]->bRenderingVertexNormals = !(pworld->_colchecker.getSurfaces()[i]->bRenderingVertexNormals);
		//	pworld->_colchecker.getSurfaces()[i]->bRenderingFaceNormals = !(pworld->_colchecker.getSurfaces()[i]->bRenderingFaceNormals);
		//}
		if ( bctrlpressed ) { // Ctrl + n = show/hide face/vertex normals of the hand geometry
			for (size_t i=0; i<pworld->_phand->_pbodies_new.size(); i++) {
				for (size_t j=0; j<pworld->_phand->_pbodies_new[i]->pSurfs.size(); j++) {
					if ( pworld->_phand->_pbodies_new[i]->pSurfs[j]->_bCollision ) {
						pworld->_phand->_pbodies_new[i]->pSurfs[j]->bRenderingFaceNormals = !(pworld->_phand->_pbodies_new[i]->pSurfs[j]->bRenderingFaceNormals);
						pworld->_phand->_pbodies_new[i]->pSurfs[j]->bRenderingVertexNormals = !(pworld->_phand->_pbodies_new[i]->pSurfs[j]->bRenderingVertexNormals);
					}
				}
			}
		} else { // n = show/hide face/vertex normals of the object
			for (size_t i=0; i<pworld->_pobject->_pSurfs.size(); i++) {
				if ( pworld->_pobject->_pSurfs[i]->_bCollision ) {
					pworld->_pobject->_pSurfs[i]->bRenderingFaceNormals = !(pworld->_pobject->_pSurfs[i]->bRenderingFaceNormals);
					pworld->_pobject->_pSurfs[i]->bRenderingVertexNormals = !(pworld->_pobject->_pSurfs[i]->bRenderingVertexNormals);
				}
			}
		}
		break;
	case 'o': // open file
		if ( bctrlpressed ) { // Ctrl + o : open grasp set file
			// open grasp set file
			filepath.clear();
			filefilternames.push_back("Grasp Set Files (*.gst)");
			filefilterpatterns.push_back("*.gst");
			filefilternames.push_back("All Files (*.*)");
			filefilterpatterns.push_back("*.*");
			OpenFileDialog(filepath, true, "Open Grasp Set File", filefilternames, filefilterpatterns);
			if ( filepath.size() > 0 ) {
				if ( !graspset.load(filepath.c_str()) ) {
					cerr << "error:: failed in loading file: " << filepath << endl;
				} else {
					cout << "a grasp set (" << graspset.size() << " grasps) loaded from " << filepath << endl;
				}
			}
			current_grasp_idx = -1;
			// check duplicate
			dup = graspset.find_duplicates(1E-6);
			if ( dup.size() > 0 ) {
				cout << "warning:: the grasp set has " << dup.size() << " duplicates!" << endl;
				cout << "          you can remove them by pressing \'w\'." << endl;
			}
		}
		else if ( bshiftpressed ) { // Shift + o : open model xml file
			// open multiple simulation data files
			filepath.clear();
			filefilternames.push_back("xml Files (*.xml)");
			filefilterpatterns.push_back("*.xml");
			filefilternames.push_back("All Files (*.*)");
			filefilterpatterns.push_back("*.*");
			OpenFileDialog(filepath, true, "Open Model XML File", filefilternames, filefilterpatterns);
			if ( filepath.size() > 0 ) {
				LoadModel(filepath.c_str());
			}
		}
		//else if ( baltpressed ) { // Alt + o : open play list file
		//	// open play list file
		//	filepath.clear();
		//	filefilternames.push_back("plt Files (*.plt)");
		//	filefilterpatterns.push_back("*.plt");
		//	filefilternames.push_back("All Files (*.*)");
		//	filefilterpatterns.push_back("*.*");
		//	OpenFileDialog(filepath, true, "Open Play List File", filefilternames, filefilterpatterns);
		//	if ( filepath.size() > 0 ) {
		//		filepaths.clear();
		//		string dirpath = ExtractDirectoryPath(filepath);
		//		ifstream fin;
		//		fin.open(filepath.c_str());
		//		getline(fin, strtmp);
		//		while (fin) {
		//			strtmp.insert(0, dirpath);
		//			filepaths.push_back(strtmp);
		//			getline(fin, strtmp);
		//		}
		//		fin.close();
		//		if ( filepaths.size() > 0 ) {
		//			LoadSimulDataFromFiles(filepaths);
		//		}
		//	}
		//}
		else if ( baltpressed ) { // Alt + o : open state file
			filepath.clear();
			filefilternames.push_back("sta Files (*.sta)");
			filefilterpatterns.push_back("*.sta");
			filefilternames.push_back("All Files (*.*)");
			filefilterpatterns.push_back("*.*");
			OpenFileDialog(filepath, true, "Open State File", filefilternames, filefilterpatterns);
			if ( filepath.size() > 0 ) {
				pworld->restoreState(filepath.c_str());
				cout << "state loaded from " << filepath << endl;
			}
		}
		else {
			// o : open simulation data files
			filepaths.clear();
			filefilternames.push_back("sim Files (*.sim)");
			filefilterpatterns.push_back("*.sim");
			filefilternames.push_back("All Files (*.*)");
			filefilterpatterns.push_back("*.*");
			OpenFileDialogMultiSelection(filepaths, "Open Simulation Data File(s)", filefilternames, filefilterpatterns);
			if ( filepaths.size() > 0 ) {
				LoadSimulDataFromFiles(filepaths);
			}
		}
		break;
	case 'p':
		if ( bctrlpressed ) {
			vector<int> picked = graspset.pick(30);
			GraspSet graspset_new;
			for (size_t i=0; i<picked.size(); i++) {
				graspset_new._grasps.push_back(graspset._grasps[picked[i]]);
			}
			graspset = graspset_new;
			cout << picked.size() << " farthest grasps picked!" << endl;
		}
		break;
	case 'r': // reset time and prepare a new simulation
		ResetTime();
		pworld->initSimulation();
		break;
	case 's': // save file
		if ( bctrlpressed ) { // Ctrl + s
			// save grasp set to file
			filepath.clear();
			filefilternames.push_back("Grasp Set Files (*.gst)");
			filefilterpatterns.push_back("*.gst");
			filefilternames.push_back("All Files (*.*)");
			filefilterpatterns.push_back("*.*");
			OpenFileDialog(filepath, false, "Save Grasp Set As", filefilternames, filefilterpatterns, "gst");
			if ( filepath.size() > 0 ) {
				graspset.save(filepath.c_str());
				cout << "the grasp set (" << graspset.size() << " grasps) saved to " << filepath << endl;
			}
		} 
		else if ( baltpressed ) { // Alt + s : save state to file
			filepath.clear();
			filefilternames.push_back("sta Files (*.sta)");
			filefilterpatterns.push_back("*.sta");
			filefilternames.push_back("All Files (*.*)");
			filefilterpatterns.push_back("*.*");
			OpenFileDialog(filepath, false, "Save State As", filefilternames, filefilterpatterns);
			if ( filepath.size() > 0 ) {
				pworld->saveState(filepath.c_str());
				cout << "current state saved to " << filepath << endl;
			}
		}
		else {
			// save simulation data to file
			filepath.clear();
			filefilternames.push_back("sim Files (*.sim)");
			filefilterpatterns.push_back("*.sim");
			filefilternames.push_back("All Files (*.*)");
			filefilterpatterns.push_back("*.*");
			//OpenFileDialog(filepath, false, "Save Simulation Data As", "sim Files (*.sim)\0*.sim\0All Files (*.*)\0*.*\0", "sim");
			OpenFileDialog(filepath, false, "Save Simulation Data As", filefilternames, filefilterpatterns, "sim");
			if ( filepath.size() > 0 ) {
				pworld->_simuldata.set_filepath_for_writing(filepath.c_str());
				pworld->_simuldata.save_data_on_memory_into_file();
				cout << "simulation data saved to " << filepath << endl;
			}
		}
		break;
	case 't': // test function
		TestFunc();
		break;
	case 'u': // show/hide usage
		bshowusage = !bshowusage;
		break;
	case 'w': // remove duplicate grasps
		dup = graspset.find_duplicates(1E-6);
		if ( dup.size() > 0 ) {
			graspset.remove(dup);
			cout << dup.size() << " duplicate grasps removed!" << endl;
		}
		break;
	case 'x': 
		bshowtextinfo = !bshowtextinfo;
		break;
	case 'y':
		pworld->_b_show_coord_frames = !(pworld->_b_show_coord_frames);
		break;
	} 

	glutPostRedisplay(); 
}

void SpecialKeyboardCallback(int key, int x, int y) 
{
	if ( !pworld ) return;

	string folderpath, filepath;
	vector<string> filefilternames, filefilterpatterns, filepaths;

	if (glutGetModifiers() == GLUT_ACTIVE_CTRL) { 
		bctrlpressed = true;
	}
	if ( glutGetModifiers() == GLUT_ACTIVE_SHIFT) {
		bshiftpressed = true;
	}
	if ( glutGetModifiers() == GLUT_ACTIVE_ALT) {
		baltpressed = true;
	}

	switch (key) {
	case GLUT_KEY_RIGHT: // right: step forward
		if ( !breplay && !bsimul ) {
			if ( bctrlpressed ) {
				current_grasp_idx++;
				if ( current_grasp_idx >= graspset.size() ) {
					current_grasp_idx = graspset.size()-1;
				}
				ApplyGrasp(current_grasp_idx);
			} 
			else if ( baltpressed ) {
				if ( bshowfingerprint ) {
					current_fingerprint_idx++;
					if ( current_fingerprint_idx >= pworld->_fingerprinthistory.size() ) {
						current_fingerprint_idx = pworld->_fingerprinthistory.size()-1;
					}
				}
			}
			else {
				current_frame_idx += 1 + num_skip_frames;
				if ( current_frame_idx >= pworld->_simuldata.size() ) {
					current_frame_idx = pworld->_simuldata.size()-1;
				}
				ReadSimulData();
			}
		}
		glutPostRedisplay(); 
		break;
	case GLUT_KEY_LEFT: // left: step backward
		if ( !breplay && !bsimul ) {
			if ( bctrlpressed ) {
				current_grasp_idx--;
				if ( current_grasp_idx < 0 ) {
					current_grasp_idx = 0;
				}
				ApplyGrasp(current_grasp_idx);
			} 
			else if ( baltpressed ) {
				if ( bshowfingerprint ) {
					current_fingerprint_idx--;
					if ( current_fingerprint_idx < 0 ) {
						current_fingerprint_idx = 0;
					}
				}
			}
			else {
				current_frame_idx -= 1 + num_skip_frames;
				if ( current_frame_idx < 0 ) {
					current_frame_idx = 0;
				}
				ReadSimulData();
			}
		}
		glutPostRedisplay(); 
		break;
	case GLUT_KEY_DOWN: 
		if ( bctrlpressed ) { // ctrl+down: decrease base refersh rate for rendering
			base_render_refersh_rate *= 0.5;
			if ( base_render_refersh_rate < 30 ) {
				base_render_refersh_rate = 30;
			}
		} else if ( bshiftpressed ) { // shift+down: scale down force and moment (rendering)
			pworld->_force_scale *= 0.5; pworld->_moment_scale *= 0.5;
			for (size_t i=0; i<pworld->_colchecker.getSurfaces().size(); i++) {
				pworld->_colchecker.getSurfaces()[i]->_force_scale *= 0.5;
			}
		} else if ( baltpressed ) { // alt+up: decrease alpha for rendering all grasps
			if ( bshowallgrasps ) {
				alpha_hand_for_rendering_all_grasps -= 0.1;
				if ( alpha_hand_for_rendering_all_grasps < 0 ) {
					alpha_hand_for_rendering_all_grasps = 0;
				}
				SetAlphaRendering(alpha_hand_for_rendering_all_grasps, 1.0);
				//cout << "alpha_hand_for_rendering_all_grasps = " << alpha_hand_for_rendering_all_grasps << endl;
			}
			if ( bshowfingerprint ) {
				fp_ref_hit_cnt_for_rendering--;
				if ( fp_ref_hit_cnt_for_rendering < 0 ) {
					fp_ref_hit_cnt_for_rendering = 0;
				}
			}
		} else { // down: replay_speed down
			// set replay speed
			replay_speed_cnt--;
			if ( replay_speed_cnt < 0 ) { replay_speed_cnt = 0; }
			replay_speed = replay_speed_set[replay_speed_cnt];
		}
		UpdateRenderRefreshRateSetting();
		glutPostRedisplay(); 
		break;
	case GLUT_KEY_UP: 
		if ( bctrlpressed ) { // ctrl+up: increase base refersh rate for rendering
			base_render_refersh_rate *= 2;
			if ( base_render_refersh_rate > 480 ) {
				base_render_refersh_rate = 480;
			}
		} else if ( bshiftpressed ) { // shift+up: scale up force and moment (rendering)
			pworld->_force_scale *= 2; pworld->_moment_scale *= 2;
			for (size_t i=0; i<pworld->_colchecker.getSurfaces().size(); i++) {
				pworld->_colchecker.getSurfaces()[i]->_force_scale *= 2;
			}
		} else if ( baltpressed ) { // alt+up: increase alpha for rendering all grasps
			if ( bshowallgrasps ) {
				alpha_hand_for_rendering_all_grasps += 0.1;
				if ( alpha_hand_for_rendering_all_grasps > 1 ) {
					alpha_hand_for_rendering_all_grasps = 1;
				}
				SetAlphaRendering(alpha_hand_for_rendering_all_grasps, 1.0);
				//cout << "alpha_hand_for_rendering_all_grasps = " << alpha_hand_for_rendering_all_grasps << endl;
			}
			if ( bshowfingerprint ) {
				fp_ref_hit_cnt_for_rendering++;
			}
		} else { // up: replay_speed up
			// set replay speed
			replay_speed_cnt++;
			if ( replay_speed_cnt >= replay_speed_set_size ) { replay_speed_cnt = replay_speed_set_size-1; }
			replay_speed = replay_speed_set[replay_speed_cnt];
		}
		UpdateRenderRefreshRateSetting();
		glutPostRedisplay(); 
		break;
	case GLUT_KEY_F1: // show manual
		cout << getManual() << endl;
		break;
	case GLUT_KEY_F2: // restore the contact saving option
		if ( bdatasavecontact != pworld->_datasavecontact ) {
			pworld->_simuldata.clear_data_on_memory();
			cout << "warning:: simulation data on memory has been cleared before restoring the contact saving option" << endl;
			pworld->_datasavecontact = bdatasavecontact;
			pworld->_setSimulationSave();
			if ( pworld->_datasavecontact ) {
				cout << "contact info will be saved during the simulation" << endl;
			} else {
				cout << "contact info will NOT be saved during the simulation" << endl;
			}
		}
		glutPostRedisplay(); 
		break;
	case GLUT_KEY_F3: // sort the current grasp set
		if ( graspset.size() == 0 ) {
			cerr << "error:: no grasp set loaded yet" << endl;
			return;
		}
		graspset.sort();
		current_grasp_idx = 0;
		cout << "grasp set sorted!" << endl;
		glutPostRedisplay(); 
		break;
	case GLUT_KEY_F4: // shuffle the grasps
		if ( graspset.size() == 0 ) {
			cerr << "error:: no grasp set loaded yet" << endl;
			return;
		}
		graspset.random_shuffle();
		current_grasp_idx = 0;
		cout << "grasp set shuffled randomly!" << endl;
		glutPostRedisplay(); 
		break;
	case GLUT_KEY_F5: // save new grasp set specified by a grasp index file
		if ( graspset.size() == 0 ) {
			cerr << "error:: no grasp set loaded yet" << endl;
			return;
		}
		// open grasp index file and save a new grasp set file
		// file format for grasp index file:
		//---------------------------
		// n
		// idx_0 idx_1 ... idx_(n-1)
		//---------------------------
		filepath.clear();
		filefilternames.clear();
		filefilterpatterns.clear();
		filefilternames.push_back("Grasp Index File (*.idx)");
		filefilterpatterns.push_back("*.idx");
		filefilternames.push_back("All Files (*.*)");
		filefilterpatterns.push_back("*.*");
		OpenFileDialog(filepath, true, "Select Grasp Index File", filefilternames, filefilterpatterns);
		if ( filepath.size() > 0 ) {
			// load grasp indices and make a new grasp set
			std::ifstream fin(filepath.c_str());
			int n; fin >> n;
			GraspSet gs;
			for (int i=0; i<n; i++) {
				int idx; fin >> idx;
				if ( idx < 0 || idx >= graspset.size() ) {
					cerr << "error:: invalid grasp index (idx = " << idx << ")" << endl;
					fin.close();
					return;
				}
				gs._grasps.push_back(graspset._grasps[idx]);
			}
			fin.close();
			// save the new grasp set
			filepath.clear();
			filefilternames.clear();
			filefilterpatterns.clear();
			filefilternames.push_back("Grasp Set Files (*.gst)");
			filefilterpatterns.push_back("*.gst");
			filefilternames.push_back("All Files (*.*)");
			filefilterpatterns.push_back("*.*");
			OpenFileDialog(filepath, false, "Save Grasp Set As", filefilternames, filefilterpatterns, "gst");
			if ( filepath.size() > 0 ) {
				gs.save(filepath.c_str());
				cout << "A new grasp set (" << gs.size() << " grasps) saved to " << filepath << endl;
			}
		}
		break;
	case GLUT_KEY_F6: // save link and object geometries to obj files
		// select a folder for saving simulation data
		OpenFolderDialog(folderpath, "Select a folder for saving geometry files (OBJ)");
		if ( folderpath.size() > 0 ) {
			for (size_t i=0; i<pworld->_pobject->getSurfaces().size(); i++) {
				stringstream ss; ss << pworld->_pobject->getName() << "_" << i << ".obj";
				string filename = folderpath; filename += "/"; filename += ss.str();
				pworld->_pobject->getSurfaces()[i]->saveToFileOBJ(filename.c_str());
			}
			vector<RigidBody*> phandlinks = pworld->_phand->getBodies();
			for (size_t i=0; i<phandlinks.size(); i++) {
				for (size_t j=0; j<phandlinks[i]->getSurfaces().size(); j++) {
					stringstream ss; ss << phandlinks[i]->getName() << "_" << j << ".obj";
					string filename = folderpath; filename += "/"; filename += ss.str();
					phandlinks[i]->getSurfaces()[j]->saveToFileOBJ(filename.c_str());
				}
			}
			cout << "obj files saved to " << folderpath << endl;
		}
		glutPostRedisplay(); 
		break;
	case GLUT_KEY_F7: // show all grasps (with rendering hands)
		if ( graspset.size() == 0 ) {
			cerr << "error:: no grasp set loaded yet" << endl;
			return;
		}
		bshowallgrasps = !bshowallgrasps;
		if ( bshowallgrasps ) {
			SetAlphaRendering(alpha_hand_for_rendering_all_grasps, 1.0);
		} else {
			SetAlphaRendering(1, 1);
		}
		glutPostRedisplay();
		break;
	case GLUT_KEY_F8: // load/show finger print
		bshowfingerprint = !bshowfingerprint;
		if ( bshowfingerprint ) {
			if ( bctrlpressed || pworld->_fingerprinthistory.size() == 0 || !pworld->checkFingerPrintHistory() ) {
				filepath.clear();
				filefilternames.clear();
				filefilterpatterns.clear();
				filefilternames.push_back("Fingerprint History File (*.fph)");
				filefilterpatterns.push_back("*.fph");
				filefilternames.push_back("All Files (*.*)");
				filefilterpatterns.push_back("*.*");
				OpenFileDialog(filepath, true, "Select Fingerprint History File", filefilternames, filefilterpatterns);
				if ( filepath.size() > 0 ) {
					if ( !pworld->_fingerprinthistory.load(filepath.c_str()) ) {
						cerr << "error:: failed in loading fingerprint history from " << filepath << endl;
						return;
					}
					current_fingerprint_idx = -1;
					fp_ref_hit_cnt_for_rendering = int(fp_ref_hit_cnt_ratio_init*float(pworld->_fingerprinthistory.size()));
					cout << "fingerprint history loaded from " << filepath << endl;
				}
			}
			if ( !pworld->checkFingerPrintHistory() ) {
				cerr << "error:: invalid fingerprint history!" << endl;
				return;
			}
		}
		glutPostRedisplay();
		break;
	case GLUT_KEY_F9: // save final grasps as a grasp set file
		if ( graspset.size() == 0 ) {
			cerr << "error:: no grasp set loaded yet" << endl;
			return;
		}
		break;
	} 
}

Vec3 get_pos_sp(int x, int y)
{
	Vec3 p;
	p[0] = 2.0 * double(x) / double(winsizeX) - 1.0;
	p[1] = 1.0 - 2.0 * double(y) / double(winsizeY);
	p[2] = p[0] * p[0] + p[1] * p[1];
	if ( p[2] < 1.0 ) p[2] = sqrt(1.0 - p[2]);
	else { p[2] = sqrt(p[2]); p[0] /= p[2]; p[1] /= p[2]; p[2] = 0.0; }
	return p;
}

void MouseCallback(int button, int state, int x, int y) 
{
	if ( !pworld ) return;

	if (glutGetModifiers() == GLUT_ACTIVE_CTRL) { 
		bctrlpressed = true;
	}
	if ( glutGetModifiers() == GLUT_ACTIVE_SHIFT) {
		bshiftpressed = true;
	}
	if ( glutGetModifiers() == GLUT_ACTIVE_ALT) {
		baltpressed = true;
	}

	downX = x; downY = y;
	bRotateView = ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN) && y < winsizeY-50 );
	bTranslateView = ((button == GLUT_MIDDLE_BUTTON) &&  (state == GLUT_DOWN));
	bZoomInOut = ((button == GLUT_RIGHT_BUTTON) &&  (state == GLUT_DOWN));
	bSliderControl = ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN) && y > winsizeY-sliderY-15);

	if ( bRotateView ) {
		mvmatrix_prev = mvmatrix;
		spoint_prev = get_pos_sp(x, y);
	}

	// wheel (zoom in/out)
	if ( (button != GLUT_LEFT_BUTTON) && (button != GLUT_MIDDLE_BUTTON) && (button != GLUT_RIGHT_BUTTON) ) {
		if ( button==3 ) { // wheel up --> zoom out
			sdepth -= 0.1f*mscale;
		} 
		if ( button==4 ) { // wheel down --> zoom in
			sdepth += 0.1f*mscale;
		}
	}

	// slider control
	if ( bSliderControl ) {
		sliderX = x; 
		if ( sliderX < sliderXi ) { sliderX = sliderXi; }
		if ( sliderX > sliderXf ) { sliderX = sliderXf; }
		if ( baltpressed ) {
			if ( bshowfingerprint && pworld->_fingerprinthistory.size() > 0 ) {
				current_fingerprint_idx = int( float(pworld->_fingerprinthistory.size()-1) * float(sliderX-sliderXi) / float(sliderXf-sliderXi) );
				if ( current_fingerprint_idx > pworld->_fingerprinthistory.size()-1 ) { current_fingerprint_idx = pworld->_fingerprinthistory.size()-1; }
				if ( current_fingerprint_idx < 0 ) { current_fingerprint_idx = 0; }
			}
		}
		else if ( bctrlpressed ) {
			if ( graspset.size() > 0 ) {
				current_grasp_idx = int( float(graspset.size()-1) * float(sliderX-sliderXi) / float(sliderXf-sliderXi) );
				if ( current_grasp_idx > graspset.size()-1 ) { current_grasp_idx = graspset.size()-1; }
				if ( current_grasp_idx < 0 ) { current_grasp_idx = 0; }
				ApplyGrasp(current_grasp_idx);
			}
		}
		else {
			current_frame_idx = int( float(pworld->_simuldata.size()-1) * float(sliderX-sliderXi) / float(sliderXf-sliderXi) );
			if ( current_frame_idx > pworld->_simuldata.size()-1 ) { current_frame_idx = pworld->_simuldata.size()-1; }
			if ( current_frame_idx < 0 ) { current_frame_idx = 0; }
			ReadSimulData();
		}
	}

	glutPostRedisplay();
}
 
void MotionCallback(int x, int y) 
{
	if ( !pworld ) return;

	// if (glutGetModifiers() == GLUT_ACTIVE_CTRL) {
	// 	bctrlpressed = true;
	// }
	// if ( glutGetModifiers() == GLUT_ACTIVE_SHIFT) {
	// 	bshiftpressed = true;
	// }
	// if ( glutGetModifiers() == GLUT_ACTIVE_ALT) {
	// 	baltpressed = true;
	// }

	if ( bRotateView ) {
		Vec3 spoint = get_pos_sp(x, y);
		double theta = acos( Inner(spoint_prev, spoint) );
		Vec3 n = Cross(spoint_prev, spoint);
		if ( Norm(n) > 1e-6 ) {
			if ( !bctrlpressed ) { n *= 3.0; }
			Vec3 w = ~(mvmatrix_prev.GetRotation()) * n;
			mvmatrix.SetRotation(mvmatrix_prev.GetRotation() * Exp(w));
		} else {
			return;
		}
	}
	if (bTranslateView) { float den = 30; if ( bctrlpressed ) { den = 300; } xcam += (float)(x-downX)/den*mscale; ycam += (float)(downY-y)/den*mscale; } // translate
	if (bZoomInOut) { float den = 30; if ( bctrlpressed ) { den = 300; } sdepth -= (float)(downY-y)/den*mscale;  } // zoom in/out
	downX = x; downY = y;

	// slider control
	if ( bSliderControl ) {
		sliderX = x; 
		if ( sliderX < sliderXi ) { sliderX = sliderXi; }
		if ( sliderX > sliderXf ) { sliderX = sliderXf; }
		if ( baltpressed ) {
			if ( bshowfingerprint && pworld->_fingerprinthistory.size() > 0 ) {
				current_fingerprint_idx = int( float(pworld->_fingerprinthistory.size()-1) * float(sliderX-sliderXi) / float(sliderXf-sliderXi) );
				if ( current_fingerprint_idx > pworld->_fingerprinthistory.size()-1 ) { current_fingerprint_idx = pworld->_fingerprinthistory.size()-1; }
				if ( current_fingerprint_idx < 0 ) { current_fingerprint_idx = 0; }
			}
		}
		else if ( bctrlpressed ) {
			if ( graspset.size() > 0 ) {
				current_grasp_idx = int( float(graspset.size()-1) * float(sliderX-sliderXi) / float(sliderXf-sliderXi) );
				if ( current_grasp_idx > graspset.size()-1 ) { current_grasp_idx = graspset.size()-1; }
				if ( current_grasp_idx < 0 ) { current_grasp_idx = 0; }
				ApplyGrasp(current_grasp_idx);
			}
		}
		else {
			current_frame_idx = int( float(pworld->_simuldata.size()-1) * float(sliderX-sliderXi) / float(sliderXf-sliderXi) );
			if ( current_frame_idx > pworld->_simuldata.size()-1 ) { current_frame_idx = pworld->_simuldata.size()-1; }
			if ( current_frame_idx < 0 ) { current_frame_idx = 0; }
			ReadSimulData();
		}
	}

	glutPostRedisplay();
}
 
void InitGL() 
{
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(640, 480);
	glutCreateWindow("gqsim");
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearColor(backgroundcolor[0],backgroundcolor[1],backgroundcolor[2],backgroundcolor[3]);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE); // call this before glEnable(GL_COLOR_MATERIAL)
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_LIGHTING);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);
	glEnable(GL_LIGHT2);
	//glEnable(GL_LIGHT3);
	//glEnable(GL_LIGHT4);
	glLightfv(GL_LIGHT0, GL_POSITION, lightpos0);
	glLightfv(GL_LIGHT1, GL_POSITION, lightpos1);
	glLightfv(GL_LIGHT2, GL_POSITION, lightpos2);
	//glLightfv(GL_LIGHT3, GL_POSITION, lightpos3);
	//glLightfv(GL_LIGHT4, GL_POSITION, lightpos4);
	glEnable(GL_CULL_FACE);
glEnable (GL_BLEND);
glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glutReshapeFunc(ReshapeCallback);
	glutDisplayFunc(DisplayCallback);
	glutKeyboardFunc(KeyboardCallback);
	glutSpecialFunc(SpecialKeyboardCallback);
	glutMouseFunc(MouseCallback);
	glutMotionFunc(MotionCallback); 
	glutTimerFunc(int(1000./base_render_refersh_rate), Timer, 0);
}

void PrintCommandLineUsage()
{
}

// parse command line input and save it to fileloadinfo
void ParseCommand(int argc, char **argv) 
{
	if ( argc == 1 ) {
		PrintCommandLineUsage();
	}

	// parse command line input
	int idx = 1;
	string cmd, filepath, filepath_graspset, folderpath_graspqualityoutput; 
	bool brungraspqualitymeasure = false, bsimulsave = true, bappend = false;
	while ( idx < argc ) {
		cmd = argv[idx];
		if ( cmd == "-help" || cmd == "-h" ) {
			PrintCommandLineUsage();
			exit(0);
		}
		else if ( cmd == "-f" || cmd == "-file" ) { // model xml file
			filepath = string(argv[++idx]);
			if ( filepath == "-fd" ) {
				vector<string> filefilternames, filefilterpatterns;
				filefilternames.push_back("XML Files (*.xml)");
				filefilterpatterns.push_back("*.xml");
				filefilternames.push_back("All Files (*.*)");
				filefilterpatterns.push_back("*.*");
				OpenFileDialog(filepath, true, "Select a file.", filefilternames, filefilterpatterns);
			}
		}
		else if ( cmd == "-g" || cmd == "-graspset" ) { // grasp set file
			filepath_graspset = string(argv[++idx]);
		}
		else if ( cmd == "-q" ) { // run measuring grasp quality and exit
			folderpath_graspqualityoutput = string(argv[++idx]);
		}
		else if ( cmd == "-app" ) { // append mode
			bappend = true;
		}
		else if ( cmd == "-nosave" || cmd == "-nosimulsave" ) { // no simulation data saving
			bsimulsave = false;
		}
		else {
			cerr << "unrecognized command flag: " << cmd << endl;
			exit(-1);
		}
		idx++;
	}

	if ( filepath.size() > 0 ) {
		LoadModel(filepath.c_str());
	}

	if ( !!pworld && filepath_graspset.size() > 0 ) {
		if (!graspset.load(filepath_graspset.c_str()) ) {
			cerr << "error:: failed in loading grasp set from " << filepath << endl;
			cout << "exit program" << endl;
			exit(0);
		}
		if ( folderpath_graspqualityoutput.size() > 0 ) {
			CreateFolder(folderpath_graspqualityoutput.c_str());
			MeasureGraspQuality(folderpath_graspqualityoutput, bsimulsave, bappend);
			cout << endl;
			cout << "grasp quality output has been saved under the folder: " << folderpath_graspqualityoutput << endl;
			cout << "exit program" << endl;
			exit(0);
		}
	}
}

int main(int argc, char **argv) 
{
	ParseCommand(argc, argv);
	glutInit(&argc, argv);
	InitGL();
	glutMainLoop(); 
	return 0; 
}
