// App.jsx
import React, { useState, useEffect } from "react";
import axios from "axios";

export default function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [datasetItems, setDatasetItems] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [showDataset, setShowDataset] = useState(false);

  // Fetch dataset items for "select outfit"
  useEffect(() => {
    if (showDataset) {
      axios.get("http://localhost:5000/api/dataset") // Backend API endpoint
        .then(res => setDatasetItems(res.data))
        .catch(err => console.error(err));
    }
  }, [showDataset]);

  // Upload file handler
  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (file) setUploadedFile(file);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("type", "uploaded");

    axios.post("http://localhost:5000/api/recommend", formData)
      .then(res => setRecommendations(res.data))
      .catch(err => console.error(err));
  };

  // Click on dataset item for recommendation
  const handleDatasetClick = (articleId) => {
    axios.post("http://localhost:5000/api/recommend", { article_id: articleId, type: "dataset" })
      .then(res => setRecommendations(res.data))
      .catch(err => console.error(err));
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <h1 className="text-3xl font-bold mb-6 text-center">Smart Outfit Recommendation</h1>

      {/* Upload Section */}
      <div className="flex justify-center mb-8">
        <label className="flex flex-col items-center justify-center w-32 h-32 bg-white rounded-full shadow-lg cursor-pointer hover:bg-gray-200">
          <span className="text-5xl font-bold text-gray-500">+</span>
          <span className="text-gray-500 text-sm mt-2">Upload</span>
          <input type="file" className="hidden" onChange={handleUpload} />
        </label>
      </div>

      {/* Select Outfits Button */}
      <div className="flex justify-center mb-8">
        <button
          className="bg-blue-500 hover:bg-blue-600 text-white font-semibold px-6 py-3 rounded"
          onClick={() => setShowDataset(!showDataset)}
        >
          {showDataset ? "Hide Outfits" : "Select Outfits"}
        </button>
      </div>

      {/* Dataset Images */}
      {showDataset && (
        <div className="grid grid-cols-3 gap-4 mb-8">
          {datasetItems.map(item => (
            <div key={item.article_id} className="bg-white rounded shadow cursor-pointer"
                 onClick={() => handleDatasetClick(item.article_id)}>
              <img
                src={`http://localhost:5000/images/${item.article_id}.jpg`}
                alt={item.product_type_name}
                className="w-full h-48 object-cover rounded-t"
              />
              <div className="p-2 text-center font-medium">{item.product_type_name}</div>
            </div>
          ))}
        </div>
      )}

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div>
          <h2 className="text-2xl font-semibold mb-4">Recommended Outfit:</h2>
          <div className="flex gap-4 flex-wrap">
            {recommendations.map((rec, idx) => (
              <div key={idx} className="bg-white rounded shadow p-2 w-48">
                <img
                  src={`http://localhost:5000/images/${rec.article_id}.jpg`}
                  alt={rec.product_type_name}
                  className="w-full h-48 object-cover rounded"
                />
                <div className="text-center mt-2 font-medium">{rec.product_type_name}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
