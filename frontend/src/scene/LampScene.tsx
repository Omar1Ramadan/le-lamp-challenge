import { useFrame } from "@react-three/fiber";
import { useRef } from "react";
import type { Group } from "three";

import type { LampPose } from "../contracts/domain";
import { neutralPose, poseToRotations } from "./pose";

interface LampSceneProps {
  pose?: LampPose;
}

function applyRotation(group: Group | null, axis: "x" | "y", value: number): void {
  if (!group) {
    return;
  }
  group.rotation[axis] += (value - group.rotation[axis]) * 0.18;
}

export function LampScene({ pose = neutralPose() }: LampSceneProps) {
  const base = useRef<Group>(null);
  const shoulder = useRef<Group>(null);
  const elbow = useRef<Group>(null);
  const wrist = useRef<Group>(null);
  const headYaw = useRef<Group>(null);
  const headPitch = useRef<Group>(null);

  useFrame(() => {
    const rotations = poseToRotations(pose);
    applyRotation(base.current, "y", rotations.base_yaw);
    applyRotation(shoulder.current, "x", rotations.shoulder_pitch);
    applyRotation(elbow.current, "x", rotations.elbow_pitch);
    applyRotation(wrist.current, "x", rotations.wrist_pitch);
    applyRotation(headYaw.current, "y", rotations.head_yaw);
    applyRotation(headPitch.current, "x", rotations.head_pitch);
  });

  return (
    <group position={[0, -1.2, 0]}>
      <mesh position={[0, -0.08, 0]}>
        <cylinderGeometry args={[0.75, 0.9, 0.16, 48]} />
        <meshStandardMaterial color="#2f3545" />
      </mesh>
      <group ref={base}>
        <group ref={shoulder} position={[0, 0.18, 0]}>
          <mesh position={[0, 0.55, 0]}>
            <boxGeometry args={[0.2, 1.1, 0.2]} />
            <meshStandardMaterial color="#7c8db5" />
          </mesh>
          <group ref={elbow} position={[0, 1.1, 0]}>
            <mesh position={[0, 0.48, 0]}>
              <boxGeometry args={[0.16, 0.95, 0.16]} />
              <meshStandardMaterial color="#9fb2db" />
            </mesh>
            <group ref={wrist} position={[0, 0.95, 0]}>
              <mesh position={[0, 0.22, 0]}>
                <boxGeometry args={[0.12, 0.45, 0.12]} />
                <meshStandardMaterial color="#b8c7e6" />
              </mesh>
              <group ref={headYaw} position={[0, 0.48, 0]}>
                <group ref={headPitch}>
                  <mesh>
                    <sphereGeometry args={[0.28, 32, 16]} />
                    <meshStandardMaterial emissive="#ff8c33" emissiveIntensity={0.45} color="#ffd0a3" />
                  </mesh>
                  <pointLight color="#ffb066" intensity={2.5} distance={4} />
                </group>
              </group>
            </group>
          </group>
        </group>
      </group>
    </group>
  );
}
