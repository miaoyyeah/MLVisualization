import React from 'react';

interface FloatingPanelProps {
  children: React.ReactNode;
}

const FloatingPanel: React.FC<FloatingPanelProps> = ({ children }) => {
  return (
    <div className="floating-panel">
      {children}
    </div>
  );
};

export default FloatingPanel;
