import React, { useState } from 'react';
import DogSimilarityVisualizer from './DogSimilarityVisualizer'; // 경로 확인
import './App.css'; // 기본 앱 CSS (옵션)

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedFile1, setSelectedFile1] = useState(null);
  const [selectedFile2, setSelectedFile2] = useState(null);
  const [previewUrl1, setPreviewUrl1] = useState(null);
  const [previewUrl2, setPreviewUrl2] = useState(null);

  // 파일 선택 핸들러
  const handleFileChange1 = (event) => {
    const file = event.target.files[0];
    setSelectedFile1(file);
    if (file) {
      setPreviewUrl1(URL.createObjectURL(file));
    } else {
      setPreviewUrl1(null);
    }
  };

  const handleFileChange2 = (event) => {
    const file = event.target.files[0];
    setSelectedFile2(file);
    if (file) {
      setPreviewUrl2(URL.createObjectURL(file));
    } else {
      setPreviewUrl2(null);
    }
  };

  // 비교 버튼 클릭 핸들러
  const handleCompare = async () => {
    if (!selectedFile1 || !selectedFile2) {
      alert("비교할 두 이미지를 선택해주세요.");
      return;
    }

    setLoading(true);
    setResult(null); // 이전 결과 초기화

    const formData = new FormData();
    formData.append('file1', selectedFile1);
    formData.append('file2', selectedFile2);

    try {
      // package.json의 proxy 설정 덕분에 상대 경로로 호출 가능
      const res = await fetch('/compare_dogs_with_heatmap/', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
        const data = await res.json();
        setResult({
            similarity: data.similarity,
            heatmapUrl1: data.heatmap_image1,
            heatmapUrl2: data.heatmap_image2,
            point1: data.point1, // 추가
            point2: data.point2, // 추가
        });
    } catch (error) {
      console.error("비교 중 오류 발생:", error);
      alert(`오류 발생: ${error.message}. 서버가 실행 중인지 확인하고, 올바른 이미지 파일을 업로드했는지 확인해주세요.`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header" style={{ marginBottom: '30px', backgroundColor: '#61dafb', padding: '20px', borderRadius: '10px' }}>
        <h1 style={{ color: 'white' }}>강아지 외형 유사도 분석기</h1>
        <p style={{ color: 'white', fontSize: '1.1em' }}>두 강아지 이미지의 외형 유사도를 비교하고, 모델이 주목한 부위를 확인해보세요!</p>
      </header>

      <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginBottom: '30px' }}>
        <div style={{ textAlign: 'center' }}>
          <input type="file" accept="image/*" onChange={handleFileChange1} />
          {previewUrl1 && <img src={previewUrl1} alt="Preview 1" style={{ maxWidth: '250px', maxHeight: '250px', marginTop: '10px', border: '1px solid #ddd', borderRadius: '5px' }} />}
        </div>
        <div style={{ textAlign: 'center' }}>
          <input type="file" accept="image/*" onChange={handleFileChange2} />
          {previewUrl2 && <img src={previewUrl2} alt="Preview 2" style={{ maxWidth: '250px', maxHeight: '250px', marginTop: '10px', border: '1px solid #ddd', borderRadius: '5px' }} />}
        </div>
      </div>

      <button
        onClick={handleCompare}
        disabled={loading || !selectedFile1 || !selectedFile2}
        style={{
          padding: '15px 30px',
          fontSize: '1.2em',
          backgroundColor: '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          cursor: 'pointer',
          transition: 'background-color 0.3s ease',
          marginBottom: '30px'
        }}
      >
        {loading ? '비교 중...' : '두 강아지 유사도 비교하기'}
      </button>

      {loading && <p>모델이 이미지를 분석하고 있습니다...</p>}

      {result && (
        <DogSimilarityVisualizer
          imageUrl1={previewUrl1} // 업로드된 이미지 미리보기 URL 사용
          imageUrl2={previewUrl2}
          heatmapUrl1={result.heatmap_image1}
          heatmapUrl2={result.heatmap_image2}
          similarityScore={result.similarity}
          point1={result.point1} // 추가
          point2={result.point2} // 추가
        />
      )}

      {!result && !loading && (
        <p style={{ marginTop: '50px', color: '#666' }}>두 강아지 이미지를 선택하고 '비교하기' 버튼을 눌러주세요.</p>
      )}
    </div>
  );
}

export default App;