import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import * as dat from 'dat.gui';

// Radial Gradient Sprite Texture
function createRadialGradientTexture(size = 64): THREE.Texture {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  // radial gradient: center = fully opaque white, edge = transparen
  const gradient = ctx.createRadialGradient(
    size / 2, size / 2, 0,
    size / 2, size / 2, size / 2
  );
  gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
  gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);

  return new THREE.CanvasTexture(canvas);
}


// Factorial
function factorial(n: number): number {
  if (n <= 1) return 1;
  let result = 1;
  for (let i = 2; i <= n; i++) {
    result *= i;
  }
  return result;
}

// Generalized Laguerre polynomial L_p^(\alpha)(x)
function laguerre(p: number, alpha: number, x: number): number {
  if (p === 0) return 1;
  if (p === 1) return 1 + alpha - x;
  let L_nm2 = 1;
  let L_nm1 = 1 + alpha - x;
  let L_n = 0;
  for (let n = 2; n <= p; n++) {
    L_n = (
      ((2 * n - 1 + alpha - x) * L_nm1 - (n - 1 + alpha) * L_nm2) / n
    );
    L_nm2 = L_nm1;
    L_nm1 = L_n;
  }
  return L_n;
}

// Associated Legendre Polynomial P_l^m(x)
function associatedLegendre(l: number, m: number, x: number): number {
  m = Math.abs(m);
  if (l === m) {
    let doubleFact = 1;
    for (let i = 1; i <= 2 * m - 1; i += 2) {
      doubleFact *= i;
    }
    return Math.pow(-1, m) * doubleFact * Math.pow(1 - x * x, m / 2);
  } else if (l === m + 1) {
    return x * (2 * m + 1) * associatedLegendre(m, m, x);
  } else {
    return (
      (
        x * (2 * l - 1) * associatedLegendre(l - 1, m, x) -
        (l + m - 1) * associatedLegendre(l - 2, m, x)
      ) /
      (l - m)
    );
  }
}

// Spherical Harmonic squared |Y_l^m(\theta,\phi)|² 
function sphericalHarmonicSquared(l: number, m: number, theta: number): number {
  const absM = Math.abs(m);
  const normalization =
    ((2 * l + 1) / (4 * Math.PI)) *
    (factorial(l - absM) / factorial(l + absM));
  const P = associatedLegendre(l, m, Math.cos(theta));
  return normalization * P * P;
}

// Radial function squared for hydrogen (a0=1)
function radialFunctionSquared(n: number, l: number, r: number): number {
  const a0 = 1;
  const rho = (2 * r) / (n * a0);
  const norm = Math.sqrt(
    Math.pow(2 / (n * a0), 3) *
      factorial(n - l - 1) /
      (2 * n * factorial(n + l))
  );
  const L = laguerre(n - l - 1, 2 * l + 1, rho);
  const R = norm * Math.pow(rho, l) * Math.exp(-rho / 2) * L;
  return R * R;
}

// Full probability density |ψₙₗₘ(r,θ,φ)|²
function wavefunctionSquared(
  n: number,
  l: number,
  m: number,
  r: number,
  theta: number,
  phi: number
): number {
  return radialFunctionSquared(n, l, r) * sphericalHarmonicSquared(l, m, theta);
}

// Probability element is |ψ|² * r² sinθ.
// estimate max by sampling random points
function estimatePmax(n: number, l: number, m: number, rMax: number): number {
  let maxP = 0;
  const samples = 500; // higher --> better accuracy
  for (let i = 0; i < samples; i++) {
    const r = Math.random() * rMax;
    const theta = Math.acos(1 - 2 * Math.random());
    const phi = Math.random() * 2 * Math.PI;
    const p = wavefunctionSquared(n, l, m, r, theta, phi) * r * r * Math.sin(theta);
    if (p > maxP) maxP = p;
  }
  return maxP;
}

// Generate positions for wavefunction>0 (+) and wavefunction<0 (-)
function generateOrbitalPointsSplit(
  n: number,
  l: number,
  m: number,
  numPoints: number,
  rMax: number
): { positionsPos: number[]; positionsNeg: number[] } {
  const positionsPos: number[] = [];
  const positionsNeg: number[] = [];

  const Pmax = estimatePmax(n, l, m, rMax);
  let count = 0;
  while (count < numPoints) {
    const r = Math.random() * rMax;
    const theta = Math.acos(1 - 2 * Math.random());
    const phi = Math.random() * 2 * Math.PI;

    // Evaluate wavefunction sign
    const R = Math.sqrt(radialFunctionSquared(n, l, r));
    const PL = associatedLegendre(l, m, Math.cos(theta));
    const cosPart = Math.cos(m * phi);
    const waveSign = R * PL * cosPart;

    // Probability element
    const P = wavefunctionSquared(n, l, m, r, theta, phi) * r * r * Math.sin(theta);
    if (Math.random() * Pmax < P) {
      const x = r * Math.sin(theta) * Math.cos(phi);
      const y = r * Math.sin(theta) * Math.sin(phi);
      const z = r * Math.cos(theta);

      if (waveSign >= 0) {
        positionsPos.push(x, y, z);
      } else {
        positionsNeg.push(x, y, z);
      }
      count++;
    }
  }
  return { positionsPos, positionsNeg };
}

//Three JS Things

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.z = 20;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Bloom composer
const composer = new EffectComposer(renderer);
const renderPass = new RenderPass(scene, camera);
composer.addPass(renderPass);

const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  1.5,  // strength
  0.4,  // radius
  0.85  // threshold
);
composer.addPass(bloomPass);


const params = {
  n: 2,
  l: 1,
  m: 0,
  numPoints: 10000,  // Higher --> smoother
  rMax: 20,
  pointSize: 0.3,
  bloomStrength: 1.5,
  bloomRadius: 0.4,
  bloomThreshold: 0.85,
  regenerate: () => generatePoints()
};

const gui = new dat.GUI();
const nController = gui.add(params, 'n', 1, 5, 1).name('n').onFinishChange(() => {
  lController.max(params.n - 1);
  if (params.l >= params.n) {
    params.l = params.n - 1;
    lController.updateDisplay();
  }
  generatePoints();
});
const lController = gui.add(params, 'l', 0, params.n - 1, 1).name('l').onFinishChange(() => {
  if (Math.abs(params.m) > params.l) {
    params.m = params.l;
  }
  generatePoints();
});
gui.add(params, 'm', -5, 5, 1).name('m').onFinishChange(() => {
  if (Math.abs(params.m) > params.l) {
    params.m = params.l;
  }
  generatePoints();
});
gui.add(params, 'numPoints', 1000, 50000, 1000).name('Number of Points').onFinishChange(() => generatePoints());
gui.add(params, 'pointSize', 0.01, 2.0, 0.01).name('Point Size').onFinishChange(() => updateMaterials());
gui.add(params, 'bloomStrength', 0, 3, 0.1).name('Bloom Strength').onChange(v => bloomPass.strength = v);
gui.add(params, 'bloomRadius', 0, 1, 0.01).name('Bloom Radius').onChange(v => bloomPass.radius = v);
gui.add(params, 'bloomThreshold', 0, 1, 0.01).name('Bloom Threshold').onChange(v => bloomPass.threshold = v);
gui.add(params, 'regenerate').name('Regenerate Points');


let spriteTexture: THREE.Texture;
let pointsPosObj: THREE.Points;
let pointsNegObj: THREE.Points;
let geometryPos: THREE.BufferGeometry;
let geometryNeg: THREE.BufferGeometry;
let materialPos: THREE.PointsMaterial;
let materialNeg: THREE.PointsMaterial;

function initSpriteTexture() {
  spriteTexture = createRadialGradientTexture(); // Soft circle sprite
}

function generatePoints() {
  if (pointsPosObj) scene.remove(pointsPosObj);
  if (pointsNegObj) scene.remove(pointsNegObj);

  const { positionsPos, positionsNeg } = generateOrbitalPointsSplit(
    params.n, params.l, params.m, params.numPoints, params.rMax
  );

  geometryPos = new THREE.BufferGeometry();
  geometryPos.setAttribute('position', new THREE.Float32BufferAttribute(positionsPos, 3));
  geometryNeg = new THREE.BufferGeometry();
  geometryNeg.setAttribute('position', new THREE.Float32BufferAttribute(positionsNeg, 3));

  materialPos = new THREE.PointsMaterial({
    map: spriteTexture,
    color: 0xff0000,
    size: params.pointSize,
    transparent: true,
    alphaTest: 0.01,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });

  materialNeg = new THREE.PointsMaterial({
    map: spriteTexture,
    color: 0x00ffff,
    size: params.pointSize,
    transparent: true,
    alphaTest: 0.01,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });

  pointsPosObj = new THREE.Points(geometryPos, materialPos);
  pointsNegObj = new THREE.Points(geometryNeg, materialNeg);

  scene.add(pointsPosObj);
  scene.add(pointsNegObj);
}

function updateMaterials() {
  if (materialPos && materialNeg) {
    materialPos.size = params.pointSize;
    materialNeg.size = params.pointSize;
  }
}


/**
 * Render loop
 */

initSpriteTexture();
generatePoints();

const controls = new OrbitControls(camera, renderer.domElement);
controls.minPolarAngle = 0;
controls.maxPolarAngle = Math.PI;

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  composer.render();
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
});
