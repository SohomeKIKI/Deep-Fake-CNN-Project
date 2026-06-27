import { useState, useRef } from 'react'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  
  const fileInputRef = useRef(null)

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0])
    }
  }

  const handleFileSelection = (selectedFile) => {
    // Check if it's an image or video
    if (selectedFile.type.startsWith('image/') || selectedFile.type.startsWith('video/')) {
      setFile(selectedFile)
      setResult(null)
      setError(null)
      
      const objectUrl = URL.createObjectURL(selectedFile)
      setPreview({
        url: objectUrl,
        type: selectedFile.type.startsWith('image/') ? 'image' : 'video'
      })
    } else {
      setError("Please upload a valid image or video file.")
    }
  }

  const handleAnalyze = async () => {
    if (!file) return

    setIsLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Failed to analyze media.")
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app-container">
      <div className="glass-card">
        <h1>Deepfake Detector</h1>
        <p className="subtitle">AI-powered media verification using CGFace.</p>

        <div 
          className={`upload-zone ${isDragging ? 'drag-active' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current.click()}
        >
          <span className="upload-icon">📂</span>
          <span className="upload-text">Drag & Drop Image or Video</span>
          <span className="upload-subtext">or click to browse files</span>
          <input 
            type="file" 
            className="file-input" 
            ref={fileInputRef}
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleFileSelection(e.target.files[0])
              }
            }}
            accept="image/*,video/*"
          />
        </div>

        {error && (
          <div style={{color: 'var(--danger)', marginTop: '1rem', fontWeight: 500}}>
            {error}
          </div>
        )}

        {preview && (
          <div className="preview-container">
            {preview.type === 'image' ? (
              <img src={preview.url} alt="Preview" className="media-preview" />
            ) : (
              <video src={preview.url} className="media-preview" controls />
            )}
            
            <div style={{marginTop: '1rem'}}>
              <button 
                className="analyze-btn"
                onClick={(e) => {
                  e.stopPropagation()
                  handleAnalyze()
                }}
                disabled={isLoading}
              >
                {isLoading ? (
                  <><span className="loader"></span> Analyzing...</>
                ) : (
                  'Analyze Media'
                )}
              </button>
            </div>
          </div>
        )}

        {result && (
          <div className="results-container">
            <div className={`prediction-badge ${result.prediction.toLowerCase()}`}>
              {result.prediction}
            </div>
            
            <div className="confidence-text">
              Confidence Score: <strong>{result.confidence}%</strong>
            </div>
            
            <div className="confidence-bar-container">
              <div 
                className={`confidence-bar ${result.prediction.toLowerCase()}`} 
                style={{ width: `${result.confidence}%` }}
              ></div>
            </div>
          </div>
        )}
        
        <div style={{ marginTop: '3rem', color: 'var(--text-secondary)', fontSize: '0.9rem', fontStyle: 'italic' }}>
          Made by Saurab Deb
        </div>
      </div>
    </div>
  )
}

export default App
