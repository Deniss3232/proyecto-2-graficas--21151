// Escena: CALLE con edificios; paso peatonal, postes de luz emisivos, arboles y un puente que une a los ultimos edificiso
// Controles: ← → rotación, ↑ ↓ altura, +/- zoom, Esc salir.

#![cfg(target_os = "windows")]

use std::f32::consts::PI;
use std::mem::{size_of, zeroed};
use std::ptr::{null, null_mut};
use std::time::Instant;

// Núcleo Raytracer 

#[derive(Clone, Copy, Debug, Default)]
struct Vec3 { x:f32, y:f32, z:f32 }
impl Vec3{
    fn new(x:f32,y:f32,z:f32)->Self{Self{x,y,z}}
    fn dot(a:Self,b:Self)->f32{a.x*b.x+a.y*b.y+a.z*b.z}
    fn cross(a:Self,b:Self)->Self{Self::new(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x)}
    fn length(self)->f32{(self.x*self.x+self.y*self.y+self.z*self.z).sqrt()}
    fn norm(self)->Self{let l=self.length().max(1e-8); self/l}
    fn reflect(v:Self,n:Self)->Self{ v - n*(2.0*Self::dot(v,n)) }
    fn refract(v:Self,n:Self,eta:f32)->Option<Self>{
        let cosi=(-Self::dot(v,n)).clamp(-1.0,1.0);
        let k=1.0-eta*eta*(1.0-cosi*cosi);
        if k<0.0{None}else{Some(v*eta+n*(eta*cosi-k.sqrt()))}
    }
}
use std::ops::{Add,Sub,Mul,Div,Neg};
impl Add for Vec3{type Output=Self;fn add(self,o:Self)->Self{Self::new(self.x+o.x,self.y+o.y,self.z+o.z)}}
impl Sub for Vec3{type Output=Self;fn sub(self,o:Self)->Self{Self::new(self.x-o.x,self.y-o.y,self.z-o.z)}}
impl Mul<f32> for Vec3{type Output=Self;fn mul(self,k:f32)->Self{Self::new(self.x*k,self.y*k,self.z*k)}}
impl Div<f32> for Vec3{type Output=Self;fn div(self,k:f32)->Self{Self::new(self.x/k,self.y/k,self.z/k)}}
impl Neg for Vec3{type Output=Self;fn neg(self)->Self{Self::new(-self.x,-self.y,-self.z)}}
impl Mul<Vec3> for Vec3{type Output=Self;fn mul(self,o:Self)->Self{Self::new(self.x*o.x,self.y*o.y,self.z*o.z)}}
fn clamp01(x:f32)->f32{x.max(0.0).min(1.0)}
fn mix3(a:Vec3,b:Vec3,t:f32)->Vec3{a*(1.0-t)+b*t}

#[derive(Clone,Copy)]struct Ray{o:Vec3,d:Vec3}

trait Texture{fn sample(&self,uv:(f32,f32),p:Vec3)->Vec3;}
struct Solid{color:Vec3} impl Texture for Solid{fn sample(&self,_:(f32,f32),_:Vec3)->Vec3{self.color}}

// ===== Ruido sencillo =====
fn hash2(u:f32,v:f32)->f32{
    let s = (u*127.1 + v*311.7).sin()*43758.5453;
    s.fract().abs()
}
fn noise2(u:f32,v:f32)->f32{
    let i = u.floor(); let j = v.floor();
    let fu = u - i; let fv = v - j;
    let a = hash2(i,   j  );
    let b = hash2(i+1.0,j  );
    let c = hash2(i,   j+1.0);
    let d = hash2(i+1.0,j+1.0);
    let s = fu*fu*(3.0-2.0*fu);
    let t = fv*fv*(3.0-2.0*fv);
    let x1 = a*(1.0-s)+b*s;
    let x2 = c*(1.0-s)+d*s;
    x1*(1.0-t)+x2*t
}

//  Texturas 

struct RoadTex { scale:f32, lane_w:f32, lane_gap:f32, zebra_v0:f32, zebra_v1:f32, zebra_w:f32 }
impl Texture for RoadTex{
    fn sample(&self, uv:(f32,f32), _p:Vec3)->Vec3{
        let (u,v)=uv;
        let base = Vec3::new(0.10,0.10,0.11) * (0.9 + 0.1*noise2(u*8.0,v*8.0));
        let du = (u-0.5).abs();
        let center = if du < self.lane_w {
            let seg = ((v*self.scale) % self.lane_gap + self.lane_gap) % self.lane_gap;
            if seg < self.lane_gap*0.4 { 1.0 } else { 0.0 }
        } else { 0.0 };
        let side = ((u-0.2).abs()<0.01 || (u-0.8).abs()<0.01) as i32 as f32;
        let zebra_band = if v>self.zebra_v0 && v<self.zebra_v1 {
            let band = ((u-self.zebra_w*0.2) / (self.zebra_w)).fract();
            (band<0.5) as i32 as f32
        } else { 0.0 };
        let paint = (center + side*0.8).clamp(0.0,1.0);
        let mix1 = mix3(base, Vec3::new(0.95,0.95,0.95), paint);
        mix3(mix1, Vec3::new(0.96,0.96,0.96), zebra_band*0.9)
    }
}

struct SidewalkTex { scale:f32 }
impl Texture for SidewalkTex{
    fn sample(&self, uv:(f32,f32), _p:Vec3)->Vec3{
        let (u,v)=uv;
        let grid_u = (u*self.scale).fract();
        let grid_v = (v*self.scale).fract();
        let seam = (grid_u<0.02 || grid_v<0.02) as i32 as f32;
        let a=Vec3::new(0.75,0.76,0.78)*(0.9+0.1*noise2(u*10.0,v*10.0));
        mix3(a, Vec3::new(0.3,0.31,0.32), seam*0.6)
    }
}

struct FacadeTex {
    wall:Vec3, frame:Vec3, glass_day:Vec3, glass_lit:Vec3,
    rows:f32, cols:f32, border:f32, light_prob:f32, seed:f32,
    base_h:f32, roof_h:f32, pilaster_every:f32
}
impl Texture for FacadeTex{
    fn sample(&self, uv:(f32,f32), _p:Vec3)->Vec3{
        let (u,v)=uv;
        let fu=(u*self.cols).fract();
        let fv=(v*self.rows).fract();

        if v > 1.0 - self.roof_h { return self.frame * (0.9+0.1*noise2(u*10.0+self.seed,v*10.0+self.seed)); }
        if v < self.base_h       { return self.wall  * (0.95+0.05*noise2(u*25.0+self.seed,v*25.0+self.seed)); }

        let pil = ((u*self.cols)/self.pilaster_every).fract();
        if pil < 0.06 { return self.frame * (0.9+0.1*noise2(u*20.0+self.seed,v*20.0+self.seed)); }

        let border= fu<self.border || fu>1.0-self.border || fv<self.border || fv>1.0-self.border;
        if border {
            return self.frame * (0.9 + 0.1*noise2(u*30.0+self.seed, v*30.0+self.seed));
        }

        let cell_u = (u*self.cols).floor();
        let cell_v = (v*self.rows).floor();
        let h = hash2(cell_u+self.seed, cell_v+self.seed);
        let on = (h < self.light_prob) as i32 as f32;
        let mut glass = mix3(self.glass_day, self.glass_lit, on);

        let sky_reflect = (1.0 - (fv-0.5).abs()*1.8).clamp(0.0,1.0)*0.28;
        glass = glass*(0.88+0.12*noise2(u*15.0+self.seed,v*15.0+self.seed)) + Vec3::new(0.5,0.6,0.9)*sky_reflect;

        let dirt = (1.0 - v).powf(2.0)*0.08;
        glass = glass*(1.0-dirt);
        Vec3::new(glass.x.max(0.09), glass.y.max(0.09), glass.z.max(0.09))
    }
}

struct ShadeHit{n:Vec3,albedo:Vec3,specular:f32,reflect:f32,transmit:f32,ior:f32,emissive:Vec3}
trait Material{fn shade(&self,uv:(f32,f32),p:Vec3,n:Vec3)->ShadeHit;}
struct Matte{tex:Box<dyn Texture>}
impl Material for Matte{fn shade(&self,uv:(f32,f32),p:Vec3,n:Vec3)->ShadeHit{
    ShadeHit{n,albedo:self.tex.sample(uv,p),specular:0.05,reflect:0.0,transmit:0.0,ior:1.0,emissive:Vec3::new(0.0,0.0,0.0)}}}
struct Metal{tex:Box<dyn Texture>,reflect:f32,specular:f32}
impl Material for Metal{fn shade(&self,uv:(f32,f32),p:Vec3,n:Vec3)->ShadeHit{
    ShadeHit{n,albedo:self.tex.sample(uv,p),specular:self.specular,reflect:self.reflect,transmit:0.0,ior:1.0,emissive:Vec3::new(0.0,0.0,0.0)}}}
struct Plastic{tex:Box<dyn Texture>,specular:f32,reflect:f32}
impl Material for Plastic{fn shade(&self,uv:(f32,f32),p:Vec3,n:Vec3)->ShadeHit{
    ShadeHit{n,albedo:self.tex.sample(uv,p),specular:self.specular,reflect:self.reflect,transmit:0.0,ior:1.0,emissive:Vec3::new(0.0,0.0,0.0)}}}
struct Glass{tint:Vec3,ior:f32,transmit:f32,reflect:f32}
impl Material for Glass{fn shade(&self,_:(f32,f32),_:Vec3,n:Vec3)->ShadeHit{
    ShadeHit{n,albedo:self.tint,specular:0.02,reflect:self.reflect,transmit:self.transmit,ior:self.ior,emissive:Vec3::new(0.0,0.0,0.0)}}}
struct Emissive{color:Vec3}
impl Material for Emissive{fn shade(&self,_:(f32,f32),_:Vec3,n:Vec3)->ShadeHit{
    ShadeHit{n,albedo:Vec3::new(0.0,0.0,0.0),specular:0.0,reflect:0.0,transmit:0.0,ior:1.0,emissive:self.color}
}}

#[derive(Clone,Copy,Default)]
struct HitRec{t:f32,p:Vec3,n:Vec3,uv:(f32,f32),mat_id:usize,hit:bool}
trait Hittable{fn hit(&self,r:&Ray,tmin:f32,tmax:f32,rec:&mut HitRec);}

struct Plane{y:f32,mat:usize,scale:f32}
impl Hittable for Plane{
    fn hit(&self,r:&Ray,tmin:f32,tmax:f32,rec:&mut HitRec){
        if r.d.y.abs()<1e-6{return;}
        let t=(self.y-r.o.y)/r.d.y; if t<tmin||t>tmax{return;}
        let p=r.o+r.d*t;
        *rec=HitRec{t,p,n:Vec3::new(0.0,1.0,0.0),uv:(p.x*self.scale,p.z*self.scale),mat_id:self.mat,hit:true};
    }
}

struct Cube{min:Vec3,max:Vec3,mat:usize}
impl Cube{fn new(min:Vec3,max:Vec3,mat:usize)->Self{Self{min,max,mat}}}
impl Hittable for Cube{
    fn hit(&self,r:&Ray,tmin:f32,tmax:f32,rec:&mut HitRec){
        let mut tminn=tmin; let mut tmaxx=tmax; let mut n=Vec3::new(0.0,0.0,0.0);
        for i in 0..3{
            let (ro,rd,mn,mx,nx,ny,nz)=match i{
                0=>(r.o.x,r.d.x,self.min.x,self.max.x,1.0,0.0,0.0),
                1=>(r.o.y,r.d.y,self.min.y,self.max.y,0.0,1.0,0.0),
                _=>(r.o.z,r.d.z,self.min.z,self.max.z,0.0,0.0,1.0),
            };
            if rd.abs()<1e-6{ if ro<mn||ro>mx{return;}}
            else{
                let (mut t1,mut t2)=((mn-ro)/rd,(mx-ro)/rd);
                let (mut n1,mut n2)=(Vec3::new(nx,ny,nz),Vec3::new(-nx,-ny,-nz));
                if t1>t2{std::mem::swap(&mut t1,&mut t2); std::mem::swap(&mut n1,&mut n2);}
                if t1>tminn{tminn=t1;n=n1;} if t2<tmaxx{tmaxx=t2;} if tminn>tmaxx{return;}
            }
        }
        if tminn<tmin||tminn>tmax{return;}
        let p=r.o+r.d*tminn;
        let (u,v)=if n.x.abs()>0.5{
            ((p.z-self.min.z)/(self.max.z-self.min.z),(p.y-self.min.y)/(self.max.y-self.min.y))
        }else if n.y.abs()>0.5{
            ((p.x-self.min.x)/(self.max.x-self.min.x),(p.z-self.min.z)/(self.max.z-self.min.z))
        }else{
            ((p.x-self.min.x)/(self.max.x-self.min.x),(p.y-self.min.y)/(self.max.y-self.min.y))
        };
        *rec=HitRec{t:tminn,p,n,uv:(u,v),mat_id:self.mat,hit:true};
    }
}

// Cielo y ambiente
fn skybox(d:Vec3)->Vec3{
    let t=0.5*(d.y+1.0);
    mix3(Vec3::new(0.92,0.95,1.0), Vec3::new(0.45,0.65,1.0), (1.0-t).clamp(0.0,1.0))
}
fn sky_ambient(n:Vec3)->Vec3{
    let up = Vec3::new(0.0,1.0,0.0);
    let k = Vec3::dot(n.norm(), up).clamp(-1.0,1.0);
    let t = (k*0.5 + 0.5).clamp(0.0,1.0);
    mix3(Vec3::new(0.2,0.22,0.25), Vec3::new(0.7,0.8,1.0), t) * 0.35
}

struct Camera{eye:Vec3,center:Vec3,up:Vec3,fov:f32,aspect:f32}
impl Camera{
    fn get_ray(&self,x:f32,y:f32)->Ray{
        let w=(self.eye-self.center).norm();
        let u=Vec3::cross(self.up,w).norm();
        let v=Vec3::cross(w,u);
        let h=(self.fov*PI/180.0*0.5).tan();
        let px=(2.0*x-1.0)*self.aspect*h;
        let py=(1.0-2.0*y)*h;
        Ray{o:self.eye,d:(u*px+v*py-w).norm()}
    }
}

struct Scene{
    objs:Vec<Box<dyn Hittable>>,
    mats:Vec<Box<dyn Material>>,
    ldir:Vec3, lcol:Vec3,
    fill_dir:Vec3, fill_col:Vec3,
    fog_color:Vec3, fog_dist:f32
}
impl Scene{
    fn trace(&self,r:&Ray,tmin:f32,tmax:f32)->HitRec{
        let mut best=HitRec::default(); let mut tb=tmax; let mut h=false;
        for o in &self.objs{ let mut tmp=HitRec::default(); o.hit(r,tmin,tb,&mut tmp); if tmp.hit&&tmp.t<tb{h=true;tb=tmp.t;best=tmp;}}
        if h{best}else{HitRec::default()}
    }
}

fn shade(s:&Scene,r:&Ray,depth:i32)->Vec3{
    if depth<=0 {return Vec3::new(0.0,0.0,0.0);}
    let rec=s.trace(r,1e-4,1e6); if !rec.hit{ return skybox(r.d); }
    let sh=s.mats[rec.mat_id].shade(rec.uv,rec.p,rec.n); let n=sh.n.norm();

    let mut col = sh.albedo * sky_ambient(n);

    let l=s.ldir.norm();
    let mut diff = sh.albedo * Vec3::dot(n,l).max(0.0);
    if s.trace(&Ray{o:rec.p+n*4e-3,d:l},1e-4,1e6).hit { diff = diff*0.5; }
    let v=(-r.d).norm(); let h=(l+v).norm();
    let spec=Vec3::new(1.0,1.0,1.0)*(Vec3::dot(n,h).max(0.0).powf(64.0))*sh.specular;
    col = col + (diff+spec)*s.lcol;

    let l2=s.fill_dir.norm();
    let diff2 = sh.albedo * (Vec3::dot(n,l2).max(0.0))*0.6;
    col = col + diff2 * s.fill_col;

    if sh.reflect>0.0{
        let rd=Vec3::reflect(r.d,n).norm();
        let refl=shade(s,&Ray{o:rec.p+n*4e-3,d:rd},depth-1);
        col=mix3(col,refl,sh.reflect);
    }
    if sh.transmit>0.0{
        let mut nout=n; let mut eta=1.0/sh.ior; let cosi=(-Vec3::dot(r.d,n)).clamp(-1.0,1.0);
        if Vec3::dot(r.d,n)>0.0{ nout=-n; eta=sh.ior; }
        if let Some(td)=Vec3::refract(r.d,nout,eta){
            let rr=shade(s,&Ray{o:rec.p-nout*4e-3,d:td.norm()},depth-1)*sh.albedo;
            col=mix3(col,rr,sh.transmit);
        }
        let r0=((1.0-sh.ior)/(1.0+sh.ior)).powi(2);
        let fr=r0+(1.0-r0)*(1.0-cosi).powi(5);
        col=mix3(col,Vec3::new(1.0,1.0,1.0),fr*0.15);
    }

    col = col + sh.emissive;

    let t = (rec.t / s.fog_dist).clamp(0.0,1.0);
    col = mix3(col, s.fog_color, t*0.6);

    Vec3::new(clamp01(col.x),clamp01(col.y),clamp01(col.z))
}

fn render(scene:&Scene, cam:&Camera, w:usize, h:usize, fb:&mut [u32]){
    for y in 0..h{
        for x in 0..w{
            let u=(x as f32+0.5)/(w as f32);
            let v=(y as f32+0.5)/(h as f32);
            let c=shade(scene,&cam.get_ray(u,v),4);
            let r=(c.x.powf(1.0/2.2)*255.0).round() as u32;
            let g=(c.y.powf(1.0/2.2)*255.0).round() as u32;
            let b=(c.z.powf(1.0/2.2)*255.0).round() as u32;
            fb[y*w+x]=(b<<16)|(g<<8)|r;
        }
    }
}

//  Escena 

fn build_scene()->Scene{
    let mut mats:Vec<Box<dyn Material>>=Vec::new();

    // Banqueta
    mats.push(Box::new(Matte{tex:Box::new(SidewalkTex{scale:18.0})}));
    // Asfalto con zebra
    mats.push(Box::new(Matte{tex:Box::new(RoadTex{
        scale:3.0, lane_w:0.02, lane_gap:1.2,
        zebra_v0:0.33, zebra_v1:0.38, zebra_w:0.08
    })}));

    // Fachadas
    let palettes = [
        (Vec3::new(0.82,0.84,0.88), Vec3::new(0.70,0.72,0.76), Vec3::new(0.55,0.78,0.90)),
        (Vec3::new(0.73,0.76,0.80), Vec3::new(0.62,0.65,0.69), Vec3::new(0.52,0.74,0.88)),
        (Vec3::new(0.66,0.69,0.73), Vec3::new(0.55,0.58,0.62), Vec3::new(0.50,0.72,0.86)),
        (Vec3::new(0.78,0.80,0.85), Vec3::new(0.67,0.70,0.75), Vec3::new(0.57,0.80,0.92)),
    ];
    for (k,(wall,frame,glass)) in palettes.iter().enumerate(){
        mats.push(Box::new(Matte{tex:Box::new(FacadeTex{
            wall:*wall, frame:*frame,
            glass_day:*glass*0.95, glass_lit:Vec3::new(1.0,0.93,0.65),
            rows: 8.0 + (k as f32), cols: 6.0 + (k as f32)*0.5,
            border:0.08, light_prob: 0.12 + 0.03*(k as f32),
            seed: 11.11*(k as f32 + 1.0),
            base_h:0.16, roof_h:0.06, pilaster_every:3.0
        })}));
    }
    //  Metal /  Plástico
    mats.push(Box::new(Metal{tex:Box::new(Solid{color:Vec3::new(0.82,0.83,0.86)}),reflect:0.55,specular:0.9}));
    mats.push(Box::new(Plastic{tex:Box::new(Solid{color:Vec3::new(0.9,0.35,0.20)}),specular:0.35,reflect:0.12}));
    //  Tronco /  Hojas
    mats.push(Box::new(Matte{ tex: Box::new(Solid{ color: Vec3::new(0.40, 0.26, 0.14) }) }));
    mats.push(Box::new(Matte{ tex: Box::new(Solid{ color: Vec3::new(0.24, 0.45, 0.22) }) }));
    // Poste gris /  Señal verde
    mats.push(Box::new(Matte{ tex: Box::new(Solid{ color: Vec3::new(0.55, 0.56, 0.60) }) }));
    mats.push(Box::new(Matte{ tex: Box::new(Solid{ color: Vec3::new(0.20, 0.55, 0.25) }) }));
    //  Emisivo
    mats.push(Box::new(Emissive{ color: Vec3::new(1.2, 1.1, 0.9) }));
    //  Vidrio (refracción)
    mats.push(Box::new(Glass{ tint: Vec3::new(0.98, 0.99, 1.0), ior: 1.5, transmit: 0.85, reflect: 0.05 }));
    let mat_glass = 13usize;

    let mut objs:Vec<Box<dyn Hittable>>=Vec::new();

    // Piso
    objs.push(Box::new(Plane{y:0.0,mat:0,scale:2.5}));

    // Calzada y banquetas
    objs.push(Box::new(Cube::new(Vec3::new(-1.6,0.01,-60.0), Vec3::new(1.6,0.03,10.0), 1)));
    objs.push(Box::new(Cube::new(Vec3::new(-3.2,0.03,-60.0), Vec3::new(-1.6,0.10,10.0), 0)));
    objs.push(Box::new(Cube::new(Vec3::new( 1.6,0.03,-60.0), Vec3::new( 3.2,0.10,10.0), 0)));

    // Edificios a lo largo
    for i in 0..10 {
        let zc = -5.0 - (i as f32)*6.0;
        let h_l = 2.6 + (i % 3) as f32 * 0.9;
        let h_r = 2.9 + ((i+1) % 4) as f32 * 0.8;
        let mid_l = 2 + (i % 4) as usize;
        let mid_r = 2 + ((i+1) % 4) as usize;

        objs.push(Box::new(Cube::new(Vec3::new(-6.0,0.0,zc-2.2), Vec3::new(-3.2,h_l,zc+2.2), mid_l)));
        objs.push(Box::new(Cube::new(Vec3::new( 3.2,0.0,zc-2.2), Vec3::new( 6.0,h_r,zc+2.2), mid_r)));

        objs.push(Box::new(Cube::new(Vec3::new(-3.5,1.1,zc-1.8), Vec3::new(-3.2,1.2,zc+1.8), 6)));
        objs.push(Box::new(Cube::new(Vec3::new(-3.4,1.0,zc-1.0), Vec3::new(-3.2,1.05,zc+1.0), 7)));
        objs.push(Box::new(Cube::new(Vec3::new( 3.2,1.1,zc-1.8), Vec3::new( 3.5,1.2,zc+1.8), 6)));
        objs.push(Box::new(Cube::new(Vec3::new( 3.2,1.0,zc-1.0), Vec3::new( 3.4,1.05,zc+1.0), 7)));
    }

    // Árboles, postes y señales
    for k in 0..6 {
        let z = -8.0 - (k as f32)*12.0;

        objs.push(Box::new(Cube::new(Vec3::new(-2.7,0.10,z-0.2), Vec3::new(-2.55,1.0,z+0.2), 8)));
        objs.push(Box::new(Cube::new(Vec3::new(-2.95,1.0,z-0.6), Vec3::new(-2.30,1.8,z+0.6), 9)));

        objs.push(Box::new(Cube::new(Vec3::new( 2.55,0.10,z-0.2), Vec3::new( 2.70,1.0,z+0.2), 8)));
        objs.push(Box::new(Cube::new(Vec3::new( 2.30,1.0,z-0.6), Vec3::new( 2.95,1.8,z+0.6), 9)));

        objs.push(Box::new(Cube::new(Vec3::new(-2.3,0.10,z-0.05), Vec3::new(-2.22,1.6,z+0.05), 10)));
        objs.push(Box::new(Cube::new(Vec3::new(-2.22,1.5,z-0.20), Vec3::new(-2.00,1.55,z+0.20), 10)));
        objs.push(Box::new(Cube::new(Vec3::new(-2.00,1.48,z-0.12), Vec3::new(-1.85,1.62,z+0.12), 12)));

        objs.push(Box::new(Cube::new(Vec3::new( 2.22,0.10,z-0.05), Vec3::new( 2.30,1.6,z+0.05), 10)));
        objs.push(Box::new(Cube::new(Vec3::new( 2.00,1.5,z-0.20), Vec3::new( 2.22,1.55,z+0.20), 10)));
        objs.push(Box::new(Cube::new(Vec3::new( 1.85,1.48,z-0.12), Vec3::new( 2.00,1.62,z+0.12), 12)));

        objs.push(Box::new(Cube::new(Vec3::new( 2.85,0.10,z-0.03), Vec3::new( 2.89,1.4,z+0.03), 10)));
        objs.push(Box::new(Cube::new(Vec3::new( 2.72,1.20,z-0.35), Vec3::new( 2.89,1.35,z+0.35), 11)));
    }

    // vitrina de vidrio 
    objs.push(Box::new(Cube::new(
        Vec3::new(2.10, 0.10, -14.6),
        Vec3::new(2.90, 1.60, -13.6),
        mat_glass
    )));

    // EDIFICIO-PUENTE 
    let z_end = -58.5f32;
    // Torres laterales
    objs.push(Box::new(Cube::new(
        Vec3::new(-6.5, 0.0, z_end-1.2),
        Vec3::new(-3.3, 3.6, z_end+1.2),
        4 // fachada
    )));
    objs.push(Box::new(Cube::new(
        Vec3::new( 3.3, 0.0, z_end-1.2),
        Vec3::new( 6.5, 3.6, z_end+1.2),
        5 
    )));
    // Puente sobre la calle
    objs.push(Box::new(Cube::new(
        Vec3::new(-3.3, 2.2, z_end-1.0),
        Vec3::new( 3.3, 3.2, z_end+1.0),
        3 
    )));
    objs.push(Box::new(Cube::new(
        Vec3::new(-1.6, 0.10, z_end+0.8),
        Vec3::new( 1.6, 0.25, z_end+0.9),
        10 
    )));

    Scene{
        objs, mats,
        ldir: Vec3::new(-0.6,-1.0,-0.25),
        lcol: Vec3::new(1.0, 0.96, 0.92)*1.8,
        fill_dir: Vec3::new(0.35,-0.6,0.45),
        fill_col: Vec3::new(0.6,0.75,1.0)*0.6,
        fog_color: Vec3::new(0.92,0.95,1.0),
        fog_dist: 80.0
    }
}

//  Win32

#[link(name="user32")] extern "system" {}
#[link(name="gdi32")]  extern "system" {}

type HINSTANCE = *mut core::ffi::c_void;
type HBRUSH    = *mut core::ffi::c_void;
type HCURSOR   = *mut core::ffi::c_void;
type HICON     = *mut core::ffi::c_void;
type HWND      = *mut core::ffi::c_void;
type HDC       = *mut core::ffi::c_void;
type HBITMAP   = *mut core::ffi::c_void;
type LPVOID    = *mut core::ffi::c_void;
type LPCWSTR   = *const u16;
type UINT      = u32;
type WPARAM    = usize;
type LPARAM    = isize;
type LRESULT   = isize;
type BOOL      = i32;

#[repr(C)] struct WNDCLASSW{style:UINT,lpfnWndProc:extern "system" fn(HWND,UINT,WPARAM,LPARAM)->LRESULT,cbClsExtra:i32,cbWndExtra:i32,hInstance:HINSTANCE,hIcon:HICON,hCursor:HCURSOR,hbrBackground:HBRUSH,lpszMenuName:LPCWSTR,lpszClassName:LPCWSTR}
#[repr(C)] struct MSG{hwnd:HWND,message:UINT,wParam:WPARAM,lParam:LPARAM,time:u32,pt:POINT}
#[repr(C)] struct POINT{ x:i32, y:i32 }
#[repr(C)] struct RECT{left:i32,top:i32,right:i32,bottom:i32}
#[repr(C)] struct BITMAPINFOHEADER{biSize:u32,biWidth:i32,biHeight:i32,biPlanes:u16,biBitCount:u16,biCompression:u32,biSizeImage:u32,biXPelsPerMeter:i32,biYPelsPerMeter:i32,biClrUsed:u32,biClrImportant:u32}
#[repr(C)] struct RGBQUAD{rgbBlue:u8,rgbGreen:u8,rgbRed:u8,rgbReserved:u8}
#[repr(C)] struct BITMAPINFO{bmiHeader:BITMAPINFOHEADER,bmiColors:[RGBQUAD;1]}
#[repr(C)] struct PAINTSTRUCT{ hdc:HDC, fErase:BOOL, rcPaint:RECT, fRestore:BOOL, fIncUpdate:BOOL, rgbReserved:[u8;32] }

const WM_DESTROY:UINT=0x0002;
const WM_PAINT:UINT  =0x000F;
const WM_SIZE:UINT   =0x0005;
const WM_TIMER:UINT  =0x0113;
const WM_KEYDOWN:UINT=0x0100;

const VK_ESCAPE:WPARAM=0x1B;
const VK_LEFT:WPARAM=0x25;
const VK_UP:WPARAM=0x26;
const VK_RIGHT:WPARAM=0x27;
const VK_DOWN:WPARAM=0x28;
const VK_ADD:WPARAM=0x6B;
const VK_SUBTRACT:WPARAM=0x6D;

const WS_OVERLAPPEDWINDOW: u32 = 0x00CF0000;
const SW_SHOW: i32 = 5;
const BI_RGB:u32=0;
const SRCCOPY:u32 = 0x00CC0020;
const GWLP_USERDATA: i32 = -21;

extern "system"{
    fn GetModuleHandleW(lpModuleName:LPCWSTR)->HINSTANCE;
    fn RegisterClassW(lpWndClass:*const WNDCLASSW)->u16;
    fn CreateWindowExW(dwExStyle:u32, lpClassName:LPCWSTR, lpWindowName:LPCWSTR, dwStyle:u32,
        X:i32,Y:i32,nWidth:i32,nHeight:i32,hWndParent:HWND,hMenu:HWND,hInstance:HINSTANCE,lpParam:LPVOID)->HWND;
    fn DefWindowProcW(hWnd:HWND,Msg:UINT,wParam:WPARAM,lParam:LPARAM)->LRESULT;
    fn ShowWindow(hWnd:HWND,nCmdShow:i32)->BOOL;
    fn UpdateWindow(hWnd:HWND)->BOOL;
    fn GetClientRect(hWnd:HWND,lpRect:*mut RECT)->BOOL;
    fn BeginPaint(hWnd:HWND,lpPaint:*mut PAINTSTRUCT)->HDC;
    fn EndPaint(hWnd:HWND,lpPaint:*const PAINTSTRUCT)->BOOL;
    fn InvalidateRect(hWnd:HWND,lpRect:*const RECT,bErase:BOOL)->BOOL;
    fn SetTimer(hWnd:HWND,nIDEvent:usize,uElapse:UINT,lpTimerFunc:LPVOID)->usize;
    fn KillTimer(hWnd:HWND,nIDEvent:usize)->BOOL;
    fn GetMessageW(lpMsg:*mut MSG,hWnd:HWND,wMsgFilterMin:UINT,wMsgFilterMax:UINT)->BOOL;
    fn TranslateMessage(lpMsg:*const MSG)->BOOL;
    fn DispatchMessageW(lpMsg:*const MSG)->isize;
    fn PostQuitMessage(nExitCode:i32);

    fn CreateCompatibleDC(hdc:HDC)->HDC;
    fn SelectObject(hdc:HDC,hgdiobj:HBITMAP)->HBITMAP;
    fn StretchBlt(hdc:HDC,x:i32,y:i32,cx:i32,cy:i32,hdcSrc:HDC,x1:i32,y1:i32,cx1:i32,cy1:i32,rop:u32)->BOOL;
    fn CreateDIBSection(hdc:HDC, pbmi:*const BITMAPINFO, iUsage:u32, ppvBits:*mut *mut core::ffi::c_void, hSection:*mut core::ffi::c_void, dwOffset:u32) -> *mut core::ffi::c_void;

    fn GetDC(hWnd:HWND)->HDC;
    fn ReleaseDC(hWnd:HWND, hDC:HDC)->i32;

    fn GetWindowLongPtrW(hWnd: HWND, nIndex: i32) -> isize;
    fn SetWindowLongPtrW(hWnd: HWND, nIndex: i32, dwNewLong: isize) -> isize;
}

//  App + WinMain 

struct App{
    scene: Scene,
    width: usize,
    height: usize,
    fb_ptr: *mut u32,
    dib: *mut core::ffi::c_void,
    memdc: *mut core::ffi::c_void,
    angle: f32,
    height_cam: f32,
    zoom: f32,
    orbit: bool,
    _t0: Instant,
}

unsafe fn app_new(hwnd:HWND, w:usize, h:usize)->App{
    let hdc = GetDC(hwnd);

    let mut bmi:BITMAPINFO = zeroed();
    bmi.bmiHeader = BITMAPINFOHEADER{
        biSize:size_of::<BITMAPINFOHEADER>() as u32,
        biWidth:w as i32, biHeight:-(h as i32),
        biPlanes:1, biBitCount:32, biCompression:BI_RGB, biSizeImage:0,
        biXPelsPerMeter:2835, biYPelsPerMeter:2835, biClrUsed:0, biClrImportant:0
    };
    let mut bits: *mut core::ffi::c_void = null_mut();
    let dib = CreateDIBSection(hdc, &bmi as *const _, 0, &mut bits, null_mut(), 0);
    let memdc = CreateCompatibleDC(hdc);
    SelectObject(memdc, dib);
    ReleaseDC(hwnd, hdc);

    App{
        scene: build_scene(),
        width:w, height:h,
        fb_ptr: bits as *mut u32,
        dib, memdc,
        angle: 0.0,
        height_cam: 3.0,
        zoom: 12.0,
        orbit: true,
        _t0: Instant::now(),
    }
}

unsafe fn app_render(app:&mut App){
    let eye = Vec3::new(app.angle.cos()*app.zoom, app.height_cam, app.angle.sin()*app.zoom + 8.0);
    let cam = Camera{
        eye,
        center:Vec3::new(0.0, 1.3, -20.0),
        up:Vec3::new(0.0,1.0,0.0),
        fov: 40.0,
        aspect: app.width as f32 / app.height as f32
    };
    let fb = std::slice::from_raw_parts_mut(app.fb_ptr, app.width*app.height);
    render(&app.scene, &cam, app.width, app.height, fb);
}

unsafe fn win_set_userdata(hwnd: HWND, p: *mut core::ffi::c_void){ let _ = SetWindowLongPtrW(hwnd, -21, p as isize); }
unsafe fn win_get_userdata(hwnd: HWND) -> *mut core::ffi::c_void { GetWindowLongPtrW(hwnd, -21) as *mut _ }

extern "system" fn wndproc(hwnd:HWND,msg:UINT,w:WPARAM,l:LPARAM)->LRESULT{
    unsafe{
        match msg{
            0x0005 => 0, // WM_SIZE
            0x0113 =>{   // WM_TIMER
                let app_ptr = win_get_userdata(hwnd) as *mut App;
                if !app_ptr.is_null(){
                    let app=&mut *app_ptr;
                    if app.orbit { app.angle += 0.6 * 0.016; }
                    app_render(app);
                    InvalidateRect(hwnd, null(), 0);
                }
                0
            }
            0x0100 =>{   // WM_KEYDOWN
                let app_ptr = win_get_userdata(hwnd) as *mut App;
                if !app_ptr.is_null(){
                    let app=&mut *app_ptr;
                    match w{
                        0x1B => { PostQuitMessage(0); return 0; } // ESC
                        0x25 => { app.angle -= 0.12; } // LEFT
                        0x27 => { app.angle += 0.12; } // RIGHT
                        0x26 => { app.height_cam += 0.2; } // UP
                        0x28 => { app.height_cam -= 0.2; } // DOWN
                        0x6B => { app.zoom = (app.zoom-0.3).max(2.0); } // VK_ADD
                        0x6D => { app.zoom += 0.3; } // VK_SUBTRACT
                        _ => {}
                    }
                }
                0
            }
            0x000F =>{   // WM_PAINT
                let mut ps:PAINTSTRUCT = zeroed();
                let hdc = BeginPaint(hwnd,&mut ps);
                let app_ptr = win_get_userdata(hwnd) as *mut App;
                if !app_ptr.is_null(){
                    let app=&mut *app_ptr;
                    let mut rc:RECT=zeroed(); GetClientRect(hwnd,&mut rc);
                    let cw=rc.right-rc.left; let ch=rc.bottom-rc.top;
                    StretchBlt(hdc,0,0,cw,ch, app.memdc, 0,0, app.width as i32, app.height as i32, 0x00CC0020);
                }
                EndPaint(hwnd,&ps);
                0
            }
            0x0002 =>{ PostQuitMessage(0); 0 } // WM_DESTROY
            _=> DefWindowProcW(hwnd,msg,w,l)
        }
    }
}

#[inline] fn to_wide(s:&str)->Vec<u16>{ s.encode_utf16().chain(std::iter::once(0)).collect() }

fn main(){
    unsafe{
        let hinst = GetModuleHandleW(null());
        let class_name = to_wide("RustWinTracerCls");
        let title = to_wide("Proyecto Gráficas – Calle con edificios (std-only)");

        let wc = WNDCLASSW{
            style: 0x0002|0x0001, // CS_HREDRAW|CS_VREDRAW
            lpfnWndProc: wndproc,
            cbClsExtra:0, cbWndExtra:0,
            hInstance:hinst, hIcon:null_mut(), hCursor:null_mut(),
            hbrBackground:null_mut(), lpszMenuName:null(), lpszClassName:class_name.as_ptr()
        };
        RegisterClassW(&wc);

        let hwnd = CreateWindowExW(0, class_name.as_ptr(), title.as_ptr(), 0x00CF0000,
                                   100,100, 1024, 640, null_mut(), null_mut(), hinst, null_mut());
        ShowWindow(hwnd, 5); UpdateWindow(hwnd);

        let mut app = app_new(hwnd, 960, 540);
        app_render(&mut app);
        win_set_userdata(hwnd, (&mut app as *mut App) as *mut _);
        InvalidateRect(hwnd, null(), 0);
        SetTimer(hwnd, 1, 16, null_mut());

        let mut msg:MSG = zeroed();
        while GetMessageW(&mut msg, null_mut(), 0, 0) != 0 {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        let _ = KillTimer(hwnd, 1);
    }
}
