import React from 'react';
import { useReactFlow } from "@xyflow/react";

export function SyncLabelNode({ data, id, selected }) {
  const { setSelectedLabel } = useReactFlow();

  const handleClick = () => {
    setSelectedLabel(data.label);
  };

  return (
    <div
      onClick={handleClick}
      className={"react-flow__node-default"}
      style={{
        border: selected ? '2px solid #ff0072' : '1px solid #222',
        boxShadow: selected ? '0 0 10px #ff0072' : 'none',
      }}
    >
      {data.label}
    </div>
  );
}
