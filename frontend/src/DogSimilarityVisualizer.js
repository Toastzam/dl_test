import React, { useState, useEffect } from 'react';
import './DogSimilarityVisualizer.css'; // CSS 파일 임포트
import FocusedImage from './FocusedImage';


function DogSimilarityVisualizer({ imageUrl1, imageUrl2, heatmapUrl1, heatmapUrl2, similarityScore, point1, point2}) {
  const [imageLoaded1, setImageLoaded1] = useState(false);
  const [imageLoaded2, setImageLoaded2] = useState(false);

  // === 여기에 추가 ===
  const 코중심X = 200; // 코의 중심 X좌표(px)
  const 코중심Y = 220; // 코의 중심 Y좌표(px)
  const 원반지름 = 40;
  
  // 이미지 실제 표시 크기
  const imageWidth = 350;
  const imageHeight = 350;

  // 백엔드에서 넘어온 heatmap의 크기 (예시: 512x512, 실제 값에 맞게 수정)
  const heatmapWidth = 512;
  const heatmapHeight = 512;

  // point2 좌표 변환 (point2가 있을 때만)
  let scaledX = null;
  let scaledY = null;
  if (point2) {
  scaledX = Math.round(point2.x * imageWidth / heatmapWidth);
  scaledY = Math.round(point2.y * imageHeight / heatmapHeight);
  }

  // 이미지 URL이 변경될 때마다 로드 상태 초기화
  useEffect(() => {
    setImageLoaded1(false);
    setImageLoaded2(false);
  }, [imageUrl1, imageUrl2]);


  console.log('point2:', point2);

  return (
    <div className="visualizer-container">
      <div className="image-comparison-area">
        {/* 첫 번째 이미지: 포커스 효과 */}
        <div className="image-wrapper">
          <FocusedImage
            src={imageUrl1}
            focusX={point1?.x ?? 175} // point1 좌표 사용
            focusY={point1?.y ?? 175}
            radius={원반지름}
            width={350}
            height={350}
          />
          {imageLoaded1 && heatmapUrl1 && (
            <img
              src={heatmapUrl1}
              alt="Heatmap 1"
              className="heatmap-overlay"
            />
          )}
        </div>

        {/* 두 번째 이미지: 원본 + 동그라미 */}
        <div className="image-wrapper" style={{ position: 'relative' }}>
          <img
            src={imageUrl2}
            alt="이미지 2"
            className="original-image"
            onLoad={() => setImageLoaded2(true)}
          />
          {/* 빨간 동그라미+오버레이 */}
          {point2 && (
            <div
                style={{
                position: 'absolute',
                left: scaledX - 원반지름,
                top: scaledY - 원반지름,
                width: 원반지름 * 2,
                height: 원반지름 * 2,
                border: '3px solid red',
                borderRadius: '50%',
                pointerEvents: 'none',
                boxSizing: 'border-box',
                background: 'rgba(255,255,0,0.18)',
                boxShadow: '0 0 20px 8px rgba(255,255,0,0.18)',
                zIndex: 2,
                }}
            />
            )}
        </div>
      </div>

    {/* 유사도 점수 및 시각화 */}
      <div className="similarity-score-area">
        <p className="similarity-label">유사도</p>
        <p className="similarity-value">{similarityScore !== undefined ? similarityScore.toFixed(2) : 'N/A'}</p>
        <div className="progress-bar-container">
          <div
            className="progress-bar-fill"
            style={{
              width: `${similarityScore * 100}%`,
              backgroundColor: similarityScore > 0.8 ? '#4CAF50' : similarityScore > 0.5 ? '#FFC107' : '#F44336'
            }}
          ></div>
        </div>
        <p className="score-interpretation">
          {similarityScore > 0.9 ? '매우 높은 유사성' :
           similarityScore > 0.7 ? '높은 유사성' :
           similarityScore > 0.5 ? '보통 수준의 유사성' :
           '낮은 유사성'}
        </p>
      </div>
    </div>
  );
}
export default DogSimilarityVisualizer;