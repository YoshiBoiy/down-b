import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import {
  FilesetResolver,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task";

const BONES = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [24, 26],
  [26, 28],
  [27, 31],
  [28, 32],
  [0, 11],
  [0, 12],
];

const video = document.getElementById("camera");
const sceneHost = document.getElementById("scene");
const startButton = document.getElementById("startCamera");
const cameraStatus = document.getElementById("cameraStatus");
const avatarSelect = document.getElementById("avatarSelect");
const avatarUpload = document.getElementById("avatarUpload");
const animateUploaded = document.getElementById("animateUploaded");
const uploadStatus = document.getElementById("uploadStatus");

let poseLandmarker;
let lastVideoTime = -1;
let latestLandmarks = null;
let latestPoseAction = "lost";
let latestPoseConfidence = 0;
let smoothedLandmarks = null;
let landmarkVelocity = null;
let smoothingTimestamp = null;
let previousPose = null;
let actionState = {
  action: "neutral",
  candidate: "neutral",
  candidateSince: 0,
  holdUntil: 0,
};
let score = { player: 0, opponent: 0 };
let opponentAction = "wait";
let opponentUntil = 0;
let opponentLaneOffset = 0;
let opponentTargetLaneOffset = 0;
let opponentDepthOffset = 0;
let opponentTargetDepthOffset = 0;
let currentAvatar = "procedural:scan";
let uploadedAvatars = [];

const gltfLoader = new GLTFLoader();
const builtInAvatars = [
  { id: "procedural:scan", label: "Crimson Scan Mesh" },
  { id: "procedural:jade", label: "Jade Body Mesh" },
  { id: "procedural:suit", label: "Training Suit" },
];

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
sceneHost.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x151918);

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera.position.set(0, 1.15, 5.2);
camera.lookAt(0, 0.15, 0);

const hemi = new THREE.HemisphereLight(0xffffff, 0x25322d, 2.2);
scene.add(hemi);
const key = new THREE.DirectionalLight(0xffffff, 1.8);
key.position.set(2.2, 4.5, 3.5);
scene.add(key);

const ring = new THREE.Group();
scene.add(ring);
buildRing();

const playerMesh = createBodyMesh(0x63d2a0, 0x24523f);
playerMesh.group.position.set(-1.05, 0, 0);
playerMesh.group.rotation.y = 0.22;
scene.add(playerMesh.group);

const opponentMesh = createBodyMesh(0xf08765, 0x65311f);
opponentMesh.group.position.set(1.25, 0, -0.15);
opponentMesh.group.rotation.y = -0.36;
scene.add(opponentMesh.group);
applyAvatarStyle(opponentMesh, "procedural:suit");

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function buildRing() {
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(5.4, 4.0),
    new THREE.MeshStandardMaterial({ color: 0x22302b, roughness: 0.9 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -1.15;
  ring.add(floor);

  const lineMaterial = new THREE.LineBasicMaterial({ color: 0xd6c38a });
  const points = [
    new THREE.Vector3(-2.7, -1.14, -2),
    new THREE.Vector3(2.7, -1.14, -2),
    new THREE.Vector3(2.7, -1.14, 2),
    new THREE.Vector3(-2.7, -1.14, 2),
    new THREE.Vector3(-2.7, -1.14, -2),
  ];
  ring.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), lineMaterial));
}

function createBodyMesh(color, darkColor) {
  const group = new THREE.Group();
  const skinMaterial = new THREE.MeshPhysicalMaterial({
    color,
    transparent: true,
    opacity: 0.78,
    roughness: 0.58,
    metalness: 0.0,
    clearcoat: 0.28,
    clearcoatRoughness: 0.55,
  });
  const suitMaterial = new THREE.MeshPhysicalMaterial({
    color: darkColor,
    transparent: true,
    opacity: 0.72,
    roughness: 0.66,
    metalness: 0.0,
    clearcoat: 0.16,
    clearcoatRoughness: 0.7,
  });
  const gloveMaterial = new THREE.MeshPhysicalMaterial({
    color: 0xf4f0e8,
    roughness: 0.38,
    metalness: 0.0,
    clearcoat: 0.42,
    clearcoatRoughness: 0.38,
  });

  const sphere = new THREE.SphereGeometry(1, 40, 28);
  const smallSphere = new THREE.SphereGeometry(1, 24, 16);
  const limb = new THREE.CylinderGeometry(1, 1, 1, 24, 4);
  const taperedLimb = new THREE.CylinderGeometry(0.78, 1, 1, 24, 4);
  const footShape = new THREE.CapsuleGeometry(1, 0.8, 6, 18);

  const make = (geometry, material) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    group.add(mesh);
    addWireOverlay(mesh, 0xff0a55, 0.2);
    return mesh;
  };

  const parts = {
    head: make(sphere, skinMaterial),
    neck: make(limb, skinMaterial),
    chest: make(sphere, suitMaterial),
    abdomen: make(sphere, suitMaterial),
    pelvis: make(sphere, suitMaterial),
    leftUpperArm: make(taperedLimb, suitMaterial),
    leftForearm: make(taperedLimb, suitMaterial),
    rightUpperArm: make(taperedLimb, suitMaterial),
    rightForearm: make(taperedLimb, suitMaterial),
    leftThigh: make(taperedLimb, suitMaterial),
    leftShin: make(taperedLimb, suitMaterial),
    rightThigh: make(taperedLimb, suitMaterial),
    rightShin: make(taperedLimb, suitMaterial),
    leftShoulder: make(smallSphere, skinMaterial),
    rightShoulder: make(smallSphere, skinMaterial),
    leftElbow: make(smallSphere, skinMaterial),
    rightElbow: make(smallSphere, skinMaterial),
    leftKnee: make(smallSphere, skinMaterial),
    rightKnee: make(smallSphere, skinMaterial),
    leftGlove: make(sphere, gloveMaterial),
    rightGlove: make(sphere, gloveMaterial),
    leftFoot: make(footShape, suitMaterial),
    rightFoot: make(footShape, suitMaterial),
  };
  const customModel = new THREE.Group();
  customModel.visible = false;
  group.add(customModel);
  const body = {
    group,
    parts,
    customModel,
    proceduralMeshes: Object.values(parts),
    dimensions: {
      bodyWidth: 0.23,
      customScale: 1.65,
    },
  };
  applyAvatarStyle(body, "procedural:scan");
  return body;
}

function addWireOverlay(mesh, color, opacity) {
  const wire = new THREE.Mesh(
    mesh.geometry,
    new THREE.MeshBasicMaterial({
      color,
      wireframe: true,
      transparent: true,
      opacity,
      depthWrite: false,
    })
  );
  wire.name = "surface-wire-overlay";
  mesh.add(wire);
  mesh.userData.wire = wire;
}

function applyAvatarStyle(body, avatarId) {
  currentAvatar = avatarId;
  const isCustom = avatarId.startsWith("custom:");
  body.showProcedural = !isCustom;
  body.customModel.visible = isCustom && body.customModel.children.length > 0;
  body.proceduralMeshes.forEach((mesh) => {
    mesh.visible = body.showProcedural;
  });
  if (isCustom) return;

  const styles = {
    "procedural:scan": {
      skin: 0xb4003d,
      suit: 0x8a0030,
      glove: 0xffd8e4,
      wire: 0xff2e69,
      opacity: 0.62,
      wireOpacity: 0.3,
    },
    "procedural:jade": {
      skin: 0x63d2a0,
      suit: 0x24523f,
      glove: 0xf4f0e8,
      wire: 0x97ffd0,
      opacity: 0.8,
      wireOpacity: 0.18,
    },
    "procedural:suit": {
      skin: 0xd98a66,
      suit: 0x442214,
      glove: 0xf4f0e8,
      wire: 0xf3a684,
      opacity: 0.9,
      wireOpacity: 0.12,
    },
  };
  const style = styles[avatarId] || styles["procedural:scan"];
  const skinParts = ["head", "neck", "leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftKnee", "rightKnee"];
  const gloveParts = ["leftGlove", "rightGlove"];
  Object.entries(body.parts).forEach(([name, mesh]) => {
    const material = mesh.material;
    if (skinParts.includes(name)) material.color.setHex(style.skin);
    else if (gloveParts.includes(name)) material.color.setHex(style.glove);
    else material.color.setHex(style.suit);
    material.opacity = style.opacity;
    material.transparent = style.opacity < 1;
    if (mesh.userData.wire) {
      mesh.userData.wire.material.color.setHex(style.wire);
      mesh.userData.wire.material.opacity = style.wireOpacity;
      mesh.userData.wire.visible = style.wireOpacity > 0;
    }
  });
}

function resize() {
  const bounds = sceneHost.getBoundingClientRect();
  renderer.setSize(bounds.width, bounds.height, false);
  camera.aspect = bounds.width / Math.max(bounds.height, 1);
  camera.updateProjectionMatrix();
}

function landmarkToScenePoint(landmark) {
  return new THREE.Vector3(
    -(landmark.x - 0.5) * 1.85,
    -(landmark.y - 0.56) * 2.55,
    -landmark.z * 2.1
  );
}

function smoothPoseLandmarks(rawLandmarks, now) {
  if (!rawLandmarks?.length) {
    smoothedLandmarks = null;
    landmarkVelocity = null;
    smoothingTimestamp = null;
    return null;
  }
  if (!smoothedLandmarks || smoothedLandmarks.length !== rawLandmarks.length || smoothingTimestamp === null) {
    smoothedLandmarks = rawLandmarks.map((point) => ({ ...point }));
    landmarkVelocity = rawLandmarks.map(() => ({ x: 0, y: 0, z: 0 }));
    smoothingTimestamp = now;
    return smoothedLandmarks;
  }

  const dt = Math.max(0.001, (now - smoothingTimestamp) / 1000);
  smoothingTimestamp = now;
  smoothedLandmarks = rawLandmarks.map((point, index) => {
    const previous = smoothedLandmarks[index];
    const previousVelocity = landmarkVelocity[index] || { x: 0, y: 0, z: 0 };
    const responsiveness = smoothingResponsiveness(index);
    const alpha = 1 - Math.exp(-responsiveness * dt);
    const maxStep = maxLandmarkStep(index) * Math.max(1, dt * 30);
    const dx = point.x - previous.x;
    const dy = point.y - previous.y;
    const dz = point.z - previous.z;
    const jump = Math.hypot(dx, dy, dz);
    const limited = jump > maxStep ? maxStep / jump : 1;
    const next = {
      ...point,
      x: previous.x + dx * limited * alpha,
      y: previous.y + dy * limited * alpha,
      z: previous.z + dz * limited * alpha,
      visibility: point.visibility ?? previous.visibility,
    };
    const velocityAlpha = 1 - Math.exp(-18 * dt);
    const vx = previousVelocity.x + ((next.x - previous.x) / dt - previousVelocity.x) * velocityAlpha;
    const vy = previousVelocity.y + ((next.y - previous.y) / dt - previousVelocity.y) * velocityAlpha;
    const vz = previousVelocity.z + ((next.z - previous.z) / dt - previousVelocity.z) * velocityAlpha;
    landmarkVelocity[index] = { x: vx, y: vy, z: vz };
    const lead = predictionLead(index);
    return {
      ...next,
      x: next.x + vx * lead,
      y: next.y + vy * lead,
      z: next.z + vz * lead,
    };
  });
  return smoothedLandmarks;
}

function smoothingResponsiveness(index) {
  if ([15, 16, 19, 20, 21, 22].includes(index)) return 42;
  if ([13, 14].includes(index)) return 30;
  if ([25, 26, 27, 28, 31, 32].includes(index)) return 18;
  if ([0, 11, 12, 23, 24].includes(index)) return 12;
  return 16;
}

function maxLandmarkStep(index) {
  if ([15, 16].includes(index)) return 0.16;
  if ([13, 14].includes(index)) return 0.12;
  if ([25, 26, 27, 28].includes(index)) return 0.1;
  return 0.065;
}

function predictionLead(index) {
  if ([15, 16].includes(index)) return 0.028;
  if ([13, 14].includes(index)) return 0.018;
  return 0.006;
}

function updateBodyMesh(mesh, landmarks) {
  if (!landmarks?.length) {
    mesh.group.visible = Boolean(mesh.customModel?.visible);
    resetCustomRig(mesh, 0.16);
    return;
  }
  mesh.group.visible = true;
  const points = landmarks.map(landmarkToScenePoint);
  const p = mesh.parts;
  const shoulderCenter = midpoint(points[11], points[12]);
  const hipCenter = midpoint(points[23], points[24]);
  const torsoLength = distanceVec(shoulderCenter, hipCenter);
  const bodyHeight = estimateBodyHeight(points, torsoLength);
  const measuredWidth = Math.max(distanceVec(points[11], points[12]), distanceVec(points[23], points[24])) * 0.52;
  const proportionalWidth = bodyHeight * 0.115;
  const widthTarget = clamp(Math.min(measuredWidth || proportionalWidth, proportionalWidth * 1.2), 0.16, 0.27);
  mesh.dimensions.bodyWidth += (widthTarget - mesh.dimensions.bodyWidth) * 0.04;
  const bodyWidth = mesh.dimensions.bodyWidth;
  const torsoQuat = quaternionBetween(hipCenter, shoulderCenter);
  updateCustomAvatar(mesh, points, bodyHeight, torsoLength);
  if (!mesh.showProcedural) {
    mesh.proceduralMeshes.forEach((part) => {
      part.visible = false;
    });
    return;
  }

  setEllipsoid(p.head, points[0], bodyWidth * 0.56, bodyWidth * 0.68, bodyWidth * 0.52);
  if (shoulderCenter && points[0]) {
    const neckTop = points[0].clone().lerp(shoulderCenter, 0.72);
    updateCylinder(p.neck, shoulderCenter, neckTop, bodyWidth * 0.22);
  } else {
    p.neck.visible = false;
  }

  if (shoulderCenter && hipCenter) {
    const chestCenter = shoulderCenter.clone().lerp(hipCenter, 0.28);
    const abdomenCenter = shoulderCenter.clone().lerp(hipCenter, 0.58);
    const pelvisCenter = shoulderCenter.clone().lerp(hipCenter, 0.9);
    setEllipsoid(p.chest, chestCenter, bodyWidth * 1.12, torsoLength * 0.26, bodyWidth * 0.48, torsoQuat);
    setEllipsoid(p.abdomen, abdomenCenter, bodyWidth * 0.92, torsoLength * 0.3, bodyWidth * 0.42, torsoQuat);
    setEllipsoid(p.pelvis, pelvisCenter, bodyWidth * 0.98, torsoLength * 0.2, bodyWidth * 0.45, torsoQuat);
  } else {
    p.chest.visible = false;
    p.abdomen.visible = false;
    p.pelvis.visible = false;
  }

  updateCylinder(p.leftUpperArm, points[11], points[13], bodyWidth * 0.18);
  updateCylinder(p.leftForearm, points[13], points[15], bodyWidth * 0.15);
  updateCylinder(p.rightUpperArm, points[12], points[14], bodyWidth * 0.18);
  updateCylinder(p.rightForearm, points[14], points[16], bodyWidth * 0.15);
  updateCylinder(p.leftThigh, points[23], points[25], bodyWidth * 0.22);
  updateCylinder(p.leftShin, points[25], points[27], bodyWidth * 0.18);
  updateCylinder(p.rightThigh, points[24], points[26], bodyWidth * 0.22);
  updateCylinder(p.rightShin, points[26], points[28], bodyWidth * 0.18);

  setEllipsoid(p.leftShoulder, points[11], bodyWidth * 0.22, bodyWidth * 0.22, bodyWidth * 0.22);
  setEllipsoid(p.rightShoulder, points[12], bodyWidth * 0.22, bodyWidth * 0.22, bodyWidth * 0.22);
  setEllipsoid(p.leftElbow, points[13], bodyWidth * 0.16, bodyWidth * 0.16, bodyWidth * 0.16);
  setEllipsoid(p.rightElbow, points[14], bodyWidth * 0.16, bodyWidth * 0.16, bodyWidth * 0.16);
  setEllipsoid(p.leftKnee, points[25], bodyWidth * 0.18, bodyWidth * 0.18, bodyWidth * 0.18);
  setEllipsoid(p.rightKnee, points[26], bodyWidth * 0.18, bodyWidth * 0.18, bodyWidth * 0.18);
  setEllipsoid(p.leftGlove, points[15], bodyWidth * 0.36, bodyWidth * 0.42, bodyWidth * 0.36);
  setEllipsoid(p.rightGlove, points[16], bodyWidth * 0.36, bodyWidth * 0.42, bodyWidth * 0.36);
  updateFoot(p.leftFoot, points[27], points[31] || points[29], bodyWidth);
  updateFoot(p.rightFoot, points[28], points[32] || points[30], bodyWidth);
}

function updateCustomAvatar(mesh, points, bodyHeight, torsoLength) {
  if (!mesh.customModel.children.length || !mesh.customModel.visible) return;
  const shoulderCenter = midpoint(points[11], points[12]);
  const hipCenter = midpoint(points[23], points[24]);
  if (!shoulderCenter || !hipCenter) return;
  const root = shoulderCenter.clone().lerp(hipCenter, 0.55);
  mesh.customModel.position.copy(root);
  const shoulderVector = new THREE.Vector3().subVectors(points[12], points[11]);
  const yaw = Math.atan2(shoulderVector.z, shoulderVector.x) - Math.PI / 2;
  mesh.customModel.rotation.set(0, yaw, 0);
  const scaleTarget = clamp((bodyHeight || torsoLength * 2.6) * 0.95, 1.1, 2.2);
  mesh.dimensions.customScale += (scaleTarget - mesh.dimensions.customScale) * 0.06;
  mesh.customModel.scale.setScalar(mesh.dimensions.customScale);
  const wrapper = mesh.customModel.children[0];
  if (animateUploaded.checked && wrapper?.userData?.rig && latestPoseAction !== "lost" && latestPoseConfidence > 0.45) {
    retargetRig(wrapper, points, mesh.group);
  } else {
    resetRigPose(wrapper, 0.08);
  }
}

function estimateBodyHeight(points, torsoLength) {
  const ankleCenter = midpoint(points[27], points[28]);
  const hipCenter = midpoint(points[23], points[24]);
  const shoulderCenter = midpoint(points[11], points[12]);
  const headToAnkles = points[0] && ankleCenter ? distanceVec(points[0], ankleCenter) : 0;
  const torsoEstimate = shoulderCenter && hipCenter ? distanceVec(shoulderCenter, hipCenter) * 2.65 : 0;
  return headToAnkles || torsoEstimate || torsoLength * 2.65 || 1.7;
}

function updateOpponentMesh(timeMs) {
  const t = timeMs / 1000;
  opponentLaneOffset += (opponentTargetLaneOffset - opponentLaneOffset) * 0.12;
  opponentDepthOffset += (opponentTargetDepthOffset - opponentDepthOffset) * 0.1;
  opponentMesh.group.position.x = 1.25 + opponentLaneOffset;
  opponentMesh.group.position.z = -0.15 + opponentDepthOffset;
  const fake = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  const slip = opponentAction === "slip_left" ? -0.08 : opponentAction === "slip_right" ? 0.08 : 0;
  const guard = opponentAction === "block";
  fake[0] = { x: 0.5 + slip, y: 0.22, z: -0.05 };
  fake[11] = { x: 0.42, y: 0.36, z: 0 };
  fake[12] = { x: 0.58, y: 0.36, z: 0 };
  fake[13] = { x: 0.37, y: 0.49, z: -0.02 };
  fake[14] = { x: 0.63, y: 0.49, z: -0.02 };
  fake[15] = {
    x: opponentAction === "jab" ? 0.18 : guard ? 0.47 : 0.39,
    y: opponentAction === "jab" || guard ? 0.31 : 0.42,
    z: opponentAction === "jab" ? -0.15 : 0,
  };
  fake[16] = {
    x: opponentAction === "cross" ? 0.2 : guard ? 0.53 : 0.61,
    y: opponentAction === "cross" || guard ? 0.31 : 0.42,
    z: opponentAction === "cross" ? -0.15 : 0,
  };
  fake[23] = { x: 0.44, y: 0.58, z: 0.02 };
  fake[24] = { x: 0.56, y: 0.58, z: 0.02 };
  fake[25] = { x: 0.43, y: 0.78, z: Math.sin(t) * 0.02 };
  fake[26] = { x: 0.57, y: 0.78, z: Math.cos(t) * 0.02 };
  fake[27] = { x: 0.42, y: 0.96, z: 0 };
  fake[28] = { x: 0.58, y: 0.96, z: 0 };
  fake[31] = fake[27];
  fake[32] = fake[28];
  updateBodyMesh(opponentMesh, fake);
}

function updateCylinder(mesh, start, end, radius = 0.05) {
  if (!start || !end) {
    mesh.visible = false;
    return;
  }
  mesh.visible = true;
  const diff = new THREE.Vector3().subVectors(end, start);
  const length = diff.length();
  mesh.position.copy(start).addScaledVector(diff, 0.5);
  mesh.scale.set(radius, Math.max(length, 0.001), radius);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), diff.normalize());
}

function updateFoot(mesh, ankle, toe, bodyWidth) {
  if (!ankle) {
    mesh.visible = false;
    return;
  }
  const fallback = ankle.clone().add(new THREE.Vector3(0, -bodyWidth * 0.08, bodyWidth * 0.5));
  const target = toe || fallback;
  const offset = new THREE.Vector3().subVectors(target, ankle);
  const maxLength = bodyWidth * 0.62;
  if (offset.length() > maxLength) offset.setLength(maxLength);
  if (offset.lengthSq() < 1e-6) offset.copy(new THREE.Vector3(0, -bodyWidth * 0.08, bodyWidth * 0.45));
  const end = ankle.clone().add(offset);
  updateCylinder(mesh, ankle, end, bodyWidth * 0.16);
  mesh.scale.x = bodyWidth * 0.2;
  mesh.scale.z = bodyWidth * 0.12;
}

function setEllipsoid(mesh, center, sx, sy, sz, quaternion = null) {
  if (!center) {
    mesh.visible = false;
    return;
  }
  mesh.visible = true;
  mesh.position.copy(center);
  mesh.scale.set(Math.max(sx, 0.001), Math.max(sy, 0.001), Math.max(sz, 0.001));
  if (quaternion) mesh.quaternion.copy(quaternion);
  else mesh.quaternion.identity();
}

function quaternionBetween(start, end) {
  if (!start || !end) return null;
  const diff = new THREE.Vector3().subVectors(end, start);
  if (diff.lengthSq() < 1e-8) return null;
  return new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), diff.normalize());
}

function midpoint(a, b) {
  if (!a || !b) return null;
  return new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
}

function distance(a, b) {
  if (!a || !b) return Infinity;
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.hypot(dx, dy, dz);
}

function distanceVec(a, b) {
  if (!a || !b) return 0;
  return a.distanceTo(b);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function classifyPose(landmarks, now) {
  if (!landmarks?.length) return { action: "lost", confidence: 0 };
  const leftWrist = landmarks[15];
  const rightWrist = landmarks[16];
  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];
  const nose = landmarks[0];
  const leftHip = landmarks[23];
  const rightHip = landmarks[24];
  const pelvis = {
    x: (leftHip.x + rightHip.x) / 2,
    y: (leftHip.y + rightHip.y) / 2,
    z: (leftHip.z + rightHip.z) / 2,
  };

  const leftGuard = Math.max(0, 1 - distance(leftWrist, nose) / 0.34);
  const rightGuard = Math.max(0, 1 - distance(rightWrist, nose) / 0.34);
  const leftExtension = distance(leftWrist, leftShoulder);
  const rightExtension = distance(rightWrist, rightShoulder);
  const leftReachTowardOpponent = leftShoulder.x - leftWrist.x;
  const rightReachTowardOpponent = rightShoulder.x - rightWrist.x;
  const headOffsetX = nose.x - pelvis.x;
  const headOffsetY = nose.y - pelvis.y;

  let leftSpeed = 0;
  let rightSpeed = 0;
  if (previousPose) {
    const dt = Math.max(0.001, (now - previousPose.now) / 1000);
    leftSpeed = distance(leftWrist, previousPose.leftWrist) / dt;
    rightSpeed = distance(rightWrist, previousPose.rightWrist) / dt;
  }

  const leftHeldPunch = leftReachTowardOpponent > 0.085 && leftExtension > 0.32;
  const rightHeldPunch = rightReachTowardOpponent > 0.085 && rightExtension > 0.32;
  const leftStartingPunch = leftSpeed > 0.75 && (leftExtension > 0.34 || leftReachTowardOpponent > 0.065);
  const rightStartingPunch = rightSpeed > 0.75 && (rightExtension > 0.34 || rightReachTowardOpponent > 0.065);

  previousPose = { now, leftWrist: { ...leftWrist }, rightWrist: { ...rightWrist } };

  let rawAction = "neutral";
  if (Math.abs(headOffsetX) > 0.1) rawAction = headOffsetX < 0 ? "left_slip" : "right_slip";
  else if (headOffsetY > -0.18) rawAction = "duck";
  else if (leftStartingPunch || leftHeldPunch) rawAction = "jab";
  else if (rightStartingPunch || rightHeldPunch) rawAction = "cross";
  else if (leftGuard > 0.45 && rightGuard > 0.45) rawAction = "high_guard";

  const action = stabilizeAction(rawAction, now, {
    leftHeldPunch,
    rightHeldPunch,
    leftGuard,
    rightGuard,
  });

  return {
    action,
    confidence: Math.min(1, landmarks.reduce((sum, point) => sum + (point.visibility || 0.75), 0) / landmarks.length),
    guard: { left: leftGuard, right: rightGuard },
    reachTowardOpponent: Math.max(leftReachTowardOpponent, rightReachTowardOpponent),
    debug: {
      rawAction,
      leftSpeed,
      rightSpeed,
      leftExtension,
      rightExtension,
      leftReachTowardOpponent,
      rightReachTowardOpponent,
      leftHeldPunch,
      rightHeldPunch,
    },
  };
}

function stabilizeAction(rawAction, now, features) {
  if (rawAction !== actionState.candidate) {
    actionState.candidate = rawAction;
    actionState.candidateSince = now;
  }

  const requiredMs = rawAction === "jab" || rawAction === "cross" ? 45 : 85;
  if (rawAction !== actionState.action && now - actionState.candidateSince >= requiredMs) {
    actionState.action = rawAction;
  }

  if (actionState.action === "jab") {
    if (features.leftHeldPunch) actionState.holdUntil = now + 280;
    if (now < actionState.holdUntil) return "jab";
  }
  if (actionState.action === "cross") {
    if (features.rightHeldPunch) actionState.holdUntil = now + 280;
    if (now < actionState.holdUntil) return "cross";
  }
  if (actionState.action === "high_guard") {
    if (features.leftGuard > 0.38 && features.rightGuard > 0.38) actionState.holdUntil = now + 180;
    if (now < actionState.holdUntil) return "high_guard";
  }

  if (rawAction === "neutral" && now >= actionState.holdUntil) {
    actionState.action = "neutral";
  }
  return actionState.action;
}

function chooseOpponent(playerAction, now) {
  if (now < opponentUntil) return opponentAction;
  opponentTargetLaneOffset = 0;
  opponentTargetDepthOffset = 0;
  if (playerAction === "jab" || playerAction === "cross") {
    const roll = Math.random();
    if (roll < 0.34) {
      opponentAction = "slip_left";
      opponentTargetLaneOffset = -0.34;
      opponentTargetDepthOffset = 0.06;
    } else if (roll < 0.68) {
      opponentAction = "slip_right";
      opponentTargetLaneOffset = 0.28;
      opponentTargetDepthOffset = 0.04;
    } else {
      opponentAction = "block";
      opponentTargetDepthOffset = 0.12;
    }
    opponentUntil = now + 360;
    return opponentAction;
  }
  if (Math.random() < 0.42) {
    opponentAction = Math.random() < 0.55 ? "jab" : "cross";
    opponentTargetDepthOffset = -0.2;
  } else if (Math.random() < 0.35) {
    opponentAction = "step";
    opponentTargetLaneOffset = Math.random() < 0.5 ? -0.2 : 0.2;
    opponentTargetDepthOffset = Math.random() < 0.5 ? -0.18 : 0.16;
  } else {
    opponentAction = "wait";
  }
  opponentUntil = now + 520;
  return opponentAction;
}

function scoreExchange(playerAction, opponent, guard, pose) {
  if (playerAction === "jab" || playerAction === "cross") {
    if (opponent === "slip_left" || opponent === "slip_right") {
      return ["miss", "opponent slipped off the punch line"];
    }
    if (opponent === "block") {
      return ["block", "opponent caught the punch on the guard"];
    }
    if ((pose?.reachTowardOpponent || 0) > 0.1) {
      score.player += 1;
      return ["hit", "punch crossed toward the opponent lane"];
    }
    return ["miss", "punch did not travel toward the opponent"];
  }
  if (opponent === "jab" || opponent === "cross") {
    if (playerAction === "left_slip" || playerAction === "right_slip" || playerAction === "duck") {
      return ["miss", "player moved the head out of line"];
    }
    if (guard.left > 0.45 && guard.right > 0.45) {
      return ["block", "both hands are high around the head"];
    }
    score.opponent += 1;
    return ["hit", "virtual punch landed on an open head line"];
  }
  return ["miss", ""];
}

async function refreshAvatarList(selectId = currentAvatar) {
  try {
    const response = await fetch("/api/avatars");
    const data = response.ok ? await response.json() : { avatars: [] };
    uploadedAvatars = data.avatars || [];
  } catch {
    uploadedAvatars = [];
  }
  avatarSelect.innerHTML = "";
  for (const avatar of builtInAvatars) {
    avatarSelect.appendChild(new Option(avatar.label, avatar.id));
  }
  if (uploadedAvatars.length) {
    const separator = new Option("Models", "", false, false);
    separator.disabled = true;
    avatarSelect.appendChild(separator);
  }
  for (const avatar of uploadedAvatars) {
    if (avatar.filename === "HumanModels.glb") {
      avatarSelect.appendChild(new Option("HumanModels - Man", `custom:${avatar.url}#variant=man`));
      avatarSelect.appendChild(new Option("HumanModels - Woman", `custom:${avatar.url}#variant=woman`));
    } else {
      avatarSelect.appendChild(new Option(avatar.name, `custom:${avatar.url}`));
    }
  }
  avatarSelect.value = selectId;
  if (avatarSelect.value !== selectId) avatarSelect.value = "procedural:scan";
}

async function selectAvatar(avatarId) {
  uploadStatus.textContent = "";
  if (!avatarId.startsWith("custom:")) {
    playerMesh.customModel.clear();
    applyAvatarStyle(playerMesh, avatarId);
    return;
  }
  const { url, variant } = parseCustomAvatarId(avatarId);
  uploadStatus.textContent = "Loading model...";
  try {
    const gltf = await gltfLoader.loadAsync(url);
    const model = normalizeImportedModel(gltf.scene, { variant });
    resetRigPose(model, 1);
    playerMesh.customModel.clear();
    playerMesh.customModel.add(model);
    playerMesh.customModel.position.set(0, -0.05, 0);
    playerMesh.customModel.rotation.set(0, 0.18, 0);
    playerMesh.customModel.scale.setScalar(1.65);
    playerMesh.group.visible = true;
    applyAvatarStyle(playerMesh, avatarId);
    const mapped = model.userData.rig?.mappedCount || 0;
    uploadStatus.textContent = mapped
      ? `Custom model loaded; ${mapped} tracked bones mapped`
      : "Custom model loaded, but no usable humanoid rig was found";
  } catch (error) {
    uploadStatus.textContent = `Could not load model: ${error.message}`;
    avatarSelect.value = "procedural:scan";
    applyAvatarStyle(playerMesh, "procedural:scan");
  }
}

function parseCustomAvatarId(avatarId) {
  const raw = avatarId.slice("custom:".length);
  const [url, hash = ""] = raw.split("#");
  const params = new URLSearchParams(hash);
  return { url, variant: params.get("variant") || "" };
}

function normalizeImportedModel(model, options = {}) {
  const wrapper = new THREE.Group();
  isolateAvatarVariant(model, options.variant);
  model.traverse((node) => {
    if (node.isMesh) {
      node.castShadow = true;
      node.receiveShadow = true;
      if (node.material) {
        node.material.transparent = true;
        node.material.opacity = Math.min(node.material.opacity ?? 1, 0.86);
      }
    }
  });
  wrapper.add(model);
  model.rotation.y = Math.PI;
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const height = Math.max(size.y, 0.001);
  model.position.set(-center.x, -center.y, -center.z);
  model.scale.setScalar(1 / height);

  const wireBox = new THREE.Box3().setFromObject(model);
  const wireSize = wireBox.getSize(new THREE.Vector3());
  const wire = new THREE.BoxHelper(model, 0xff2e69);
  wire.material.transparent = true;
  wire.material.opacity = wireSize.length() > 0 ? 0.18 : 0;
  wrapper.add(wire);
  wrapper.userData.rig = buildHumanoidRig(wrapper);
  return wrapper;
}

function isolateAvatarVariant(model, variant = "") {
  const skinnedMeshes = [];
  model.traverse((node) => {
    if (node.isSkinnedMesh) skinnedMeshes.push(node);
  });
  if (!skinnedMeshes.length) return;

  const selected =
    skinnedMeshes.find((node) => variant && node.name.toLowerCase().includes(variant)) ||
    skinnedMeshes.find((node) => node.parent?.name.toLowerCase().includes(variant)) ||
    skinnedMeshes[0];
  const selectedRoot = topLevelChildFor(model, selected) || selected;

  for (const child of model.children) {
    child.visible = child === selectedRoot;
  }
  selectedRoot.traverse((node) => {
    node.visible = true;
  });
}

function topLevelChildFor(root, node) {
  let current = node;
  while (current?.parent && current.parent !== root) {
    current = current.parent;
  }
  return current?.parent === root ? current : null;
}

function buildHumanoidRig(wrapper) {
  wrapper.updateMatrixWorld(true);
  const bones = [];
  wrapper.traverse((node) => {
    if (node.isBone) bones.push(node);
  });
  const byRole = {
    hips: findExactBone(bones, ["hips", "hip", "pelvis", "spine"]) || findBone(bones, ["hips", "hip", "pelvis"]),
    spine: findExactBone(bones, ["spine.002", "spine.001", "spine", "waist_04", "bust_05"]) || findBone(bones, ["spine", "waist", "bust", "chest"]),
    chest: findExactBone(bones, ["spine.004", "spine.003", "bust_05", "chest"]) || findBone(bones, ["chest", "bust", "spine2", "spine_2"]),
    neck: findExactBone(bones, ["neck", "neck_06"]) || findBone(bones, ["neck"]),
    head: findExactBone(bones, ["head", "head_07"]) || findBone(bones, ["head"]),
    leftUpperArm: findExactBone(bones, ["upper_arm.l", "mixamorigleftarm", "shoulderl_024", "arml_025"]) || findSideBone(bones, "left", ["upperarm", "upper_arm", "shoulder", "arm"]),
    leftForearm: findExactBone(bones, ["forearm.l", "mixamorigleftforearm", "arml_025", "handl_026"]) || findSideBone(bones, "left", ["forearm", "lowerarm", "arm"]),
    leftHand: findExactBone(bones, ["hand.l", "mixamoriglefthand", "handl_026"]) || findSideBone(bones, "left", ["hand", "wrist"]),
    rightUpperArm: findExactBone(bones, ["upper_arm.r", "mixamorigrightarm", "shoulderr_010", "armr_011"]) || findSideBone(bones, "right", ["upperarm", "upper_arm", "shoulder", "arm"]),
    rightForearm: findExactBone(bones, ["forearm.r", "mixamorigrightforearm", "armr_011", "handr_012"]) || findSideBone(bones, "right", ["forearm", "lowerarm", "arm"]),
    rightHand: findExactBone(bones, ["hand.r", "mixamorigRightHand", "handr_012"]) || findSideBone(bones, "right", ["hand", "wrist"]),
    leftUpperLeg: findExactBone(bones, ["thigh.l", "upper_leg.l", "mixamorigleftupleg", "legl_040"]) || findSideBone(bones, "left", ["upleg", "upperleg", "thigh", "leg"]),
    leftLowerLeg: findExactBone(bones, ["shin.l", "calf.l", "mixamorigleftleg", "kneel_041"]) || findSideBone(bones, "left", ["lowerleg", "calf", "knee"]),
    leftFoot: findExactBone(bones, ["foot.l", "mixamorigleftfoot", "footl_042"]) || findSideBone(bones, "left", ["foot", "ankle"]),
    rightUpperLeg: findExactBone(bones, ["thigh.r", "upper_leg.r", "mixamorigrightupleg", "legr_044"]) || findSideBone(bones, "right", ["upleg", "upperleg", "thigh", "leg"]),
    rightLowerLeg: findExactBone(bones, ["shin.r", "calf.r", "mixamorigrightleg", "kneer_045"]) || findSideBone(bones, "right", ["lowerleg", "calf", "knee"]),
    rightFoot: findExactBone(bones, ["foot.r", "mixamorigrightfoot", "footr_046"]) || findSideBone(bones, "right", ["foot", "ankle"]),
  };
  const segments = [
    ["spine", "hips", 23, 11],
    ["chest", "spine", 23, 11],
    ["neck", "neck", 11, 0],
    ["head", "head", 11, 0],
    ["leftUpperArm", "leftUpperArm", 11, 13],
    ["leftForearm", "leftForearm", 13, 15],
    ["rightUpperArm", "rightUpperArm", 12, 14],
    ["rightForearm", "rightForearm", 14, 16],
    ["leftUpperLeg", "leftUpperLeg", 23, 25],
    ["leftLowerLeg", "leftLowerLeg", 25, 27],
    ["rightUpperLeg", "rightUpperLeg", 24, 26],
    ["rightLowerLeg", "rightLowerLeg", 26, 28],
  ];
  const retargeters = [];
  for (const [name, role, startIndex, endIndex] of segments) {
    const bone = byRole[role];
    if (!bone) continue;
    const child = firstBoneChild(bone);
    const parent = bone.parent || wrapper;
    const start = parent.worldToLocal(bone.getWorldPosition(new THREE.Vector3()));
    const end = child
      ? parent.worldToLocal(child.getWorldPosition(new THREE.Vector3()))
      : start.clone().add(new THREE.Vector3(0, 1, 0));
    const restDir = end.sub(start).normalize();
    if (restDir.lengthSq() < 1e-8) continue;
    retargeters.push({
      name,
      bone,
      parent,
      startIndex,
      endIndex,
      restDir,
      restLocalQuaternion: bone.quaternion.clone(),
    });
  }
  return {
    bones,
    retargeters,
    mappedCount: retargeters.length,
  };
}

function findExactBone(bones, names) {
  const wanted = names.map(cleanBoneName);
  return bones.find((bone) => wanted.includes(cleanBoneName(bone.name))) || null;
}

function findBone(bones, tokens) {
  return bones.find((bone) => {
    const name = cleanBoneName(bone.name);
    return tokens.some((token) => name.includes(cleanBoneName(token))) && !name.includes("end") && !name.startsWith("h");
  });
}

function findSideBone(bones, side, tokens) {
  const sideTokens = side === "left" ? ["left", " l ", "_l", " l", "l_", "legl", "arml", "shoulderl", "handl", "kneel", "footl"] : ["right", " r ", "_r", " r", "r_", "legr", "armr", "shoulderr", "handr", "kneer", "footr"];
  const candidates = bones.filter((bone) => {
    const raw = ` ${bone.name.toLowerCase()} `;
    const name = cleanBoneName(bone.name);
    const hasSide = sideTokens.some((token) => raw.includes(token) || name.includes(cleanBoneName(token)));
    const hasPart = tokens.some((token) => name.includes(cleanBoneName(token)));
    return hasSide && hasPart && !name.includes("end") && !name.startsWith("h");
  });
  return candidates.sort((a, b) => scoreBoneName(a.name, tokens) - scoreBoneName(b.name, tokens))[0] || null;
}

function scoreBoneName(name, tokens) {
  const clean = cleanBoneName(name);
  let score = clean.length;
  tokens.forEach((token, index) => {
    const cleanedToken = cleanBoneName(token);
    if (clean === cleanedToken) score -= 200 - index * 20;
    else if (clean.includes(cleanedToken)) score -= 100 - index * 20;
  });
  if (clean.includes("helper") || clean.startsWith("h")) score += 50;
  return score;
}

function cleanBoneName(name) {
  return name.toLowerCase().replace(/mixamorig|[^a-z0-9.]/g, "");
}

function firstBoneChild(bone) {
  return bone.children.find((child) => child.isBone) || null;
}

function retargetRig(wrapper, points, bodyGroup) {
  const rig = wrapper.userData.rig;
  if (!rig?.retargeters?.length) return;
  wrapper.updateMatrixWorld(true);
  for (const item of rig.retargeters) {
    const start = points[item.startIndex];
    const end = points[item.endIndex];
    if (!start || !end) continue;
    const startWorld = bodyGroup.localToWorld(start.clone());
    const endWorld = bodyGroup.localToWorld(end.clone());
    item.parent.updateMatrixWorld(true);
    const startLocal = item.parent.worldToLocal(startWorld);
    const endLocal = item.parent.worldToLocal(endWorld);
    const targetDir = endLocal.sub(startLocal).normalize();
    if (targetDir.lengthSq() < 1e-8) continue;
    const delta = new THREE.Quaternion().setFromUnitVectors(item.restDir, targetDir);
    const targetQuaternion = item.restLocalQuaternion.clone().premultiply(delta);
    item.bone.quaternion.slerp(targetQuaternion, 0.42);
  }
}

function resetCustomRig(mesh, alpha = 0.12) {
  const wrapper = mesh.customModel?.children?.[0];
  resetRigPose(wrapper, alpha);
}

function resetRigPose(wrapper, alpha = 0.12) {
  const rig = wrapper?.userData?.rig;
  if (!rig?.retargeters?.length) return;
  for (const item of rig.retargeters) {
    item.bone.quaternion.slerp(item.restLocalQuaternion, alpha);
  }
}

async function uploadAvatar(file) {
  if (!file) return;
  uploadStatus.textContent = "Uploading model...";
  const form = new FormData();
  form.append("avatar", file);
  try {
    const response = await fetch("/api/upload-avatar", { method: "POST", body: form });
    if (!response.ok) throw new Error(await response.text());
    const avatar = await response.json();
    await refreshAvatarList(`custom:${avatar.url}`);
    await selectAvatar(`custom:${avatar.url}`);
    uploadStatus.textContent = `Stored ${avatar.filename}`;
  } catch (error) {
    uploadStatus.textContent = `Upload failed: ${error.message}`;
  } finally {
    avatarUpload.value = "";
  }
}

async function initPoseLandmarker() {
  cameraStatus.textContent = "Loading pose model...";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: POSE_MODEL_URL,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputSegmentationMasks: false,
  });
}

async function startCamera() {
  startButton.disabled = true;
  try {
    if (!poseLandmarker) await initPoseLandmarker();
    cameraStatus.textContent = "Requesting camera...";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    updateVideoPreviewOrientation();
    cameraStatus.textContent = "Tracking live webcam pose";
    requestAnimationFrame(loop);
  } catch (error) {
    cameraStatus.textContent = `Camera failed: ${error.message}`;
    startButton.disabled = false;
  }
}

function updateVideoPreviewOrientation() {
  const isPortraitStream = video.videoHeight > video.videoWidth;
  video.classList.toggle("portrait-stream", isPortraitStream);
}

function loop(now) {
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = poseLandmarker.detectForVideo(video, now);
    latestLandmarks = smoothPoseLandmarks(result.landmarks?.[0] || null, now);
    const pose = classifyPose(latestLandmarks, now);
    latestPoseAction = pose.action;
    latestPoseConfidence = pose.confidence;
    const opponent = chooseOpponent(pose.action, now);
    const [exchange, reason] = scoreExchange(pose.action, opponent, pose.guard || { left: 0, right: 0 }, pose);
    setText("playerAction", pose.action);
    setText("opponentAction", opponent);
    setText("exchange", exchange);
    setText("score", `${score.player} - ${score.opponent}`);
    setText("confidence", pose.confidence.toFixed(2));
    setText("reason", reason);
    setText("adaptation", pose.guard?.right < 0.35 ? "right guard is low after exchanges" : "");
  }
  updateBodyMesh(playerMesh, latestLandmarks);
  updateOpponentMesh(now);
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}

window.addEventListener("resize", resize);
startButton.addEventListener("click", startCamera);
avatarSelect.addEventListener("change", () => selectAvatar(avatarSelect.value));
avatarUpload.addEventListener("change", () => uploadAvatar(avatarUpload.files?.[0]));
resize();
refreshAvatarList();
renderer.setAnimationLoop((time) => {
  if (!poseLandmarker) {
    updateOpponentMesh(time);
    renderer.render(scene, camera);
  }
});
startCamera();
