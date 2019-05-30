const fs = require("fs");
const implicitMesh = require("implicit-mesh");
const serializeObj = require("serialize-wavefront-obj");
const { GPU } = require("gpu.js");

const SIZE = 128;
const NUM_POINTS = 1000;

const range = length => Array.from({ length }, (_, i) => i);

const gpu = new GPU();

// SDF buildig blocks
gpu.addFunction(function length3(x, y, z) {
  return Math.sqrt(x * x + y * y + z * z);
});

gpu.addFunction(function length2(x, y) {
  return Math.sqrt(x * x + y * y);
});

gpu.addFunction(function sphere(x, y, z, r) {
  return length3(x, y, z) - r;
});

gpu.addFunction(function unionRound(a, b, r) {
  const aa = Math.max(r - a, 0);
  const bb = Math.max(r - b, 0);

  return Math.max(r, Math.min(aa, bb)) - length2(aa, bb);
});

// random spheres
const spherePositions = range(NUM_POINTS).map(() => [
  Math.random(),
  Math.random(),
  Math.random()
]);

// generates signed distance for given x/y/z position
const kernel = gpu.createKernel(
  function kernelFunction(spherePositions, numSpheres, SIZE) {
    const r = 0.01; // sphere size
    const ur = 0.1; // union roundness

    const x = this.thread.x - SIZE / 2;
    const y = this.thread.y - SIZE / 2;
    const z = this.thread.z - SIZE / 2;

    let sd = 1.0; // signed distance

    // unioning all spheres
    for (let i = 0; i < numSpheres; i = i + 1) {
      sd = unionRound(
        sd,
        sphere(
          x / SIZE - spherePositions[i][0],
          y / SIZE - spherePositions[i][1],
          z / SIZE - spherePositions[i][2],
          r
        ),
        ur
      );
    }

    return sd;
  },
  { output: [SIZE, SIZE, SIZE] }
);

// 3-dimensional cube of signed distances
console.time("kernel calculation");
const result = kernel(spherePositions, spherePositions.length, SIZE);
console.timeEnd("kernel calculation");

// meshing to OBJ
console.time("sdf meshing");
const { cells, positions } = implicitMesh(SIZE, (x, y, z) => {
  const cx = (x / 2) * SIZE + SIZE / 2;
  const cy = (y / 2) * SIZE + SIZE / 2;
  const cz = (z / 2) * SIZE + SIZE / 2;

  return result[cx][cy][cz];
});
console.timeEnd("sdf meshing");

fs.writeFileSync("./mesh.obj", serializeObj(cells, positions), "utf-8");
console.log("done!");

