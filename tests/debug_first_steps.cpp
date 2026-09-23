#include <cmath>
#include "dpfpga/global_placer.hpp"
using namespace dpfpga;
static double l1(const std::vector<Real>& v){double s=0;for(auto x:v)s+=std::abs(double(x));return s;}
int main(int argc,char**argv){
  Params p; p.aux_input=argv[1]; p.routability_opt_flag=1;
  PlaceDB db; db.read(p); db.initialize(p);
  NonLinearPlacer placer(p,db);
  auto& pos=placer.pos();
  const int n=db.num_nodes();
  PlaceObjective model(placer.data(),p,placer.regions());
  model.initialize_density_weight(pos);
  std::printf("gamma %g\n",double(model.gamma()));
  for(int r=0;r<4;++r) std::printf("region %d density_weight %.6e\n",r,double(model.density_weight()[r]));
  // gradient pieces
  std::vector<Real> g;
  Real obj=model.obj_and_grad(pos,g);
  std::printf("obj %.6e  |g|_1 %.6e\n",double(obj),l1(g));
  // per-region: density gradient L1 (raw)
  for(int r=0;r<2;++r){
    auto& reg=*placer.regions()[r];
    Real e=reg.energy(pos.data(),true);
    std::vector<Real> gr(pos.size(),0); reg.gradient(1,pos.data(),gr.data());
    std::printf("region %d energy %.6e raw dens grad L1 %.6e\n",r,double(e),l1(gr));
  }
  // spread of movable cells
  double sx=0,sy=0,mx=0,my=0; int m=db.num_movable_nodes;
  for(int i=0;i<m;++i){mx+=pos[i];my+=pos[n+i];} mx/=m;my/=m;
  for(int i=0;i<m;++i){sx+=(pos[i]-mx)*(pos[i]-mx);sy+=(pos[n+i]-my)*(pos[n+i]-my);}
  std::printf("center (%.3f, %.3f) std (%.4f, %.4f)\n",mx,my,std::sqrt(sx/m),std::sqrt(sy/m));
  Real lr=model.estimate_initial_learning_rate(pos);
  std::printf("initial lr %.6e\n",double(lr));
  // filler stats
  double fx=0; int f0=db.num_physical_nodes; int cnt=db.num_filler_nodes;
  for(int i=0;i<cnt;++i) fx+=pos[f0+i];
  std::printf("fillers %d mean x %.3f\n",cnt,fx/cnt);
}
