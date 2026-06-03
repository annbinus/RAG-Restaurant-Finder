"use client";

import { useState } from "react";

interface Restaurant {
  name: string;
  city: string;
  state: string;
  stars: number;
  review_count: number;
  categories: string;
  score: number;
}

interface SearchResponse {
  query: string;
  answer: string;
  results: Restaurant[];
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [city, setCity] = useState("Philadelphia");
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSearch() {
    if (!query.trim()) return;
    setLoading(true);
    setResponse(null);

    const res = await fetch("http://localhost:8000/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, city }),
    });

    const data = await res.json();
    setResponse(data);
    setLoading(false);
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-white mb-2">Restaurant Finder</h1>
        <p className="text-slate-400 mb-8">RAG-powered semantic restaurant search</p>

        {/* Search bar */}
        <div className="bg-slate-700 rounded-lg p-6 mb-8">
          <div className="flex gap-2 mb-4">
            <input
              type="text"
              placeholder="great Thai food, cozy ramen, late night tacos..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              className="flex-1 bg-slate-600 text-white px-4 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={handleSearch}
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white px-6 py-3 rounded-lg font-medium"
            >
              {loading ? "Thinking..." : "Search"}
            </button>
          </div>
          <div className="flex gap-2 items-center">
            <label className="text-slate-300 text-sm">City:</label>
            <input
              type="text"
              value={city}
              onChange={(e) => setCity(e.target.value)}
              className="bg-slate-600 text-white px-3 py-1 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* AI answer */}
        {response?.answer && (
          <div className="bg-slate-700 border border-blue-500/30 rounded-lg p-6 mb-6">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-2 h-2 rounded-full bg-blue-400" />
              <span className="text-blue-400 text-sm font-medium">AI Summary</span>
            </div>
            <p className="text-slate-200 leading-relaxed">{response.answer}</p>
          </div>
        )}

        {/* Result cards */}
        {response?.results && (
          <>
            <p className="text-slate-500 text-sm mb-4">
              {response.results.length} restaurants retrieved
            </p>
            <div className="space-y-4">
              {response.results.map((r, i) => (
                <div
                  key={i}
                  className="bg-slate-700 rounded-lg p-4 hover:bg-slate-600 transition"
                >
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="text-xl font-bold text-white">{r.name}</h3>
                    <span className="text-yellow-400 text-sm">
                      {r.stars}★ ({r.review_count})
                    </span>
                  </div>
                  <p className="text-slate-400 text-sm mb-2">{r.categories}</p>
                  <div className="flex justify-between text-xs text-slate-500">
                    <span>{r.city}, {r.state}</span>
                    <span className="text-blue-400">relevance: {r.score.toFixed(3)}</span>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {!response && !loading && (
          <div className="text-center text-slate-400">
            Enter a query to search
          </div>
        )}
      </div>
    </div>
  );
}