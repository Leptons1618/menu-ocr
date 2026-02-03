import React, { useState, useEffect } from 'react';
import { FileText, BookOpen, BarChart3, Target, Cpu, ArrowRight, ExternalLink, Download, ChevronDown, ChevronUp } from 'lucide-react';

// Research findings and methodology content
const RESEARCH_SECTIONS = [
  {
    id: 'overview',
    title: 'Research Overview',
    icon: BookOpen,
    content: `Menu OCR is a modular pipeline for extracting structured data from restaurant menu images. 
    The system combines optical character recognition (OCR) with multiple classification approaches—rule-based 
    heuristics, traditional machine learning models, and ensemble methods—to produce schema-compliant JSON 
    output without hallucinating structure not present in the source image.`
  },
  {
    id: 'architecture',
    title: 'System Architecture',
    icon: Cpu,
    content: `The pipeline processes menu images in six sequential stages:

**1. OCR (Text Extraction)** - EasyOCR extracts text with extended spatial metadata including polygon coordinates, rotation degrees, and baseline positions.

**2. Column Detection** - DBSCAN clustering on x-coordinates identifies layout columns. Price columns are identified by rightmost position and high digit ratio.

**3. Classification** - A hybrid classifier uses rule-based primary classification with ML as secondary signal. Rule confidence threshold of 0.7 ensures ML only adjusts uncertain cases.

**4. Hierarchy Enforcement** - Viterbi decoding with FSM constraints ensures valid label sequences (e.g., Section→Group→Item→Price).

**5. Price Matching** - Hungarian algorithm for global bipartite matching prevents multiple items claiming the same price.

**6. Structure Building** - Assembles classified elements into hierarchical JSON output.`
  },
  {
    id: 'methodology',
    title: 'Classification Methodology',
    icon: Target,
    content: `**Rule-Based Classification** uses deterministic heuristics based on:
- Price pattern detection (regex for currency formats)
- Font size analysis via KMeans clustering  
- Positional features (normalized coordinates)
- Lexical priors (section/category keywords)
- Gap analysis (vertical spacing patterns)

**ML Classification** trained on CORD-v2 dataset (11,000 receipt images) achieves 91% accuracy on receipts but underperforms on menus due to domain mismatch.

**Hybrid Approach**: ML is only consulted when rule confidence < 0.7, with predictions discounted by 30% due to domain mismatch.

**Key Insight**: Domain-specific rule-based classification outperforms general-purpose ML trained on receipts.`
  },
  {
    id: 'results',
    title: 'Experimental Results',
    icon: BarChart3,
    content: `**Final Performance on Menu Test Set (491 items, 20 images):**

| Version | Precision | Recall | F1 | Price Acc |
|---------|-----------|--------|-----|-----------|
| v1.0 Baseline | 19.8% | 44.0% | 27.3% | 19.9% |
| v2.0 Fixed GT | 38.7% | 62.4% | 47.8% | 31.7% |
| **v2.1 Current** | **75.8%** | **70.1%** | **72.8%** | 34.8% |

**Top Performing Images:**
- menu_0003.jpg: F1=97% (Cocktails menu)
- menu_0015.jpg: F1=93% (Veg Main Course)
- menu_0012.jpg: F1=89% (Starter & Tandoor)

**Key Improvements:**
- OCR noise filtering (+10% precision)
- Fixed ground truth data (+15% F1)
- Expanded keyword dictionaries (177 section, 123 group)`
  },
  {
    id: 'improvements',
    title: 'Accuracy Improvements',
    icon: Target,
    content: `**v2.1 Enhancements:**

1. **OCR Noise Filtering** - Detect and filter mixed-case noise patterns ("RuX", "jOn"), short word whitelist

2. **Enhanced Price Detection** - 11 regex patterns for international formats (Rs, INR, €, £, ¥, price ranges)

3. **Improved Column Detection** - DBSCAN with adaptive epsilon based on document width

4. **Better Hierarchy Enforcement** - FSM with Viterbi decoding prevents invalid sequences

5. **Global Price Matching** - Hungarian algorithm replaces greedy assignment

6. **Expanded Keywords** - 177 section indicators, 123 group/category keywords

**Remaining Challenges:**
- Multi-column layouts need better reading order
- Price accuracy still limited (34.8%)
- Complex stylized menus with decorative fonts`
  }
];

const METRICS_DATA = [
  { label: 'F1 Score', value: '72.8%', color: '#22c55e' },
  { label: 'Precision', value: '75.8%', color: '#3b82f6' },
  { label: 'Recall', value: '70.1%', color: '#f59e0b' },
  { label: 'Processing Speed', value: '310ms', color: '#8b5cf6' }
];

function MetricCard({ label, value, color }) {
  return (
    <div style={{ 
      background: 'var(--color-gray-50)', 
      borderRadius: 12, 
      padding: 20,
      borderLeft: `4px solid ${color}`
    }}>
      <div style={{ fontSize: 28, fontWeight: 600, color }}>{value}</div>
      <div style={{ fontSize: 13, color: 'var(--color-gray-500)', marginTop: 4 }}>{label}</div>
    </div>
  );
}

function ResearchSection({ section, isExpanded, onToggle }) {
  const Icon = section.icon;
  
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div 
        className="card-header" 
        style={{ cursor: 'pointer' }}
        onClick={onToggle}
      >
        <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Icon size={18} />
          {section.title}
        </span>
        {isExpanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
      </div>
      {isExpanded && (
        <div className="card-body">
          <div className="markdown-content" dangerouslySetInnerHTML={{ __html: formatMarkdown(section.content) }} />
        </div>
      )}
    </div>
  );
}

function formatMarkdown(text) {
  // Simple markdown formatting
  let html = text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br/>')
    .replace(/\|(.+)\|/g, (match) => {
      const cells = match.split('|').filter(c => c.trim());
      return '<tr>' + cells.map(c => `<td style="padding: 8px 12px; border: 1px solid var(--color-gray-200);">${c.trim()}</td>`).join('') + '</tr>';
    });
  
  // Wrap tables
  if (html.includes('<tr>')) {
    html = html.replace(/(<tr>[\s\S]*?<\/tr>)+/g, '<table style="border-collapse: collapse; width: 100%; margin: 16px 0;">$&</table>');
  }
  
  return `<p>${html}</p>`;
}

export default function ResearchPage() {
  const [expandedSections, setExpandedSections] = useState(new Set(['overview']));
  const [activeTab, setActiveTab] = useState('findings');
  
  const toggleSection = (id) => {
    setExpandedSections(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };
  
  const expandAll = () => {
    setExpandedSections(new Set(RESEARCH_SECTIONS.map(s => s.id)));
  };
  
  const collapseAll = () => {
    setExpandedSections(new Set());
  };
  
  return (
    <div className="container">
      <h1 className="page-title">Research & Documentation</h1>
      <p className="page-subtitle">
        Technical paper, methodology, and experimental findings
      </p>
      
      {/* Metrics summary */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 16, marginBottom: 32 }}>
        {METRICS_DATA.map((metric, idx) => (
          <MetricCard key={idx} {...metric} />
        ))}
      </div>
      
      {/* Tabs */}
      <div className="tabs">
        <button 
          className={`tab ${activeTab === 'findings' ? 'active' : ''}`}
          onClick={() => setActiveTab('findings')}
        >
          <BarChart3 size={16} style={{ marginRight: 6 }} />
          Research Findings
        </button>
        <button 
          className={`tab ${activeTab === 'paper' ? 'active' : ''}`}
          onClick={() => setActiveTab('paper')}
        >
          <FileText size={16} style={{ marginRight: 6 }} />
          Technical Paper
        </button>
      </div>
      
      {activeTab === 'findings' && (
        <>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginBottom: 16 }}>
            <button className="btn btn-ghost" onClick={expandAll} style={{ fontSize: 12, padding: '4px 12px' }}>
              Expand All
            </button>
            <button className="btn btn-ghost" onClick={collapseAll} style={{ fontSize: 12, padding: '4px 12px' }}>
              Collapse All
            </button>
          </div>
          
          {RESEARCH_SECTIONS.map(section => (
            <ResearchSection
              key={section.id}
              section={section}
              isExpanded={expandedSections.has(section.id)}
              onToggle={() => toggleSection(section.id)}
            />
          ))}
          
          {/* Recommendations */}
          <div className="card" style={{ marginTop: 24, background: 'var(--color-gray-50)' }}>
            <div className="card-header">
              <span className="card-title">Key Recommendations</span>
            </div>
            <div className="card-body">
              <ol style={{ margin: 0, paddingLeft: 20, lineHeight: 1.8 }}>
                <li><strong>Use Rule-Based Classification</strong> when domain-specific training data is unavailable</li>
                <li><strong>Prioritize OCR Quality</strong> - detection errors propagate irrecoverably through the pipeline</li>
                <li><strong>Enable GPU Acceleration</strong> for real-time applications (3.5× speedup)</li>
                <li><strong>Use Bipartite Matching</strong> for price-item association to avoid greedy conflicts</li>
                <li><strong>Detect Columns Explicitly</strong> before resolving reading order</li>
                <li><strong>Train on Domain-Specific Data</strong> if menu-labeled datasets become available</li>
              </ol>
            </div>
          </div>
        </>
      )}
      
      {activeTab === 'paper' && (
        <div className="card">
          <div className="card-header">
            <span className="card-title">
              <FileText size={16} style={{ marginRight: 8 }} />
              Menu OCR: A Modular Pipeline for Structured Menu Extraction
            </span>
            <div style={{ display: 'flex', gap: 8 }}>
              <a 
                href="/paper/menu_ocr_paper.pdf" 
                target="_blank" 
                rel="noopener noreferrer"
                className="btn btn-primary"
                style={{ fontSize: 12, padding: '6px 12px' }}
              >
                <ExternalLink size={14} />
                Open PDF
              </a>
              <a 
                href="/paper/menu_ocr_paper.pdf" 
                download="menu_ocr_paper.pdf"
                className="btn btn-secondary"
                style={{ fontSize: 12, padding: '6px 12px' }}
              >
                <Download size={14} />
                Download
              </a>
            </div>
          </div>
          <div className="card-body" style={{ padding: 24 }}>
            {/* Paper abstract and key sections rendered as HTML */}
            <div style={{ maxWidth: 800, margin: '0 auto' }}>
              <h2 style={{ fontSize: 20, marginBottom: 16 }}>Abstract</h2>
              <p style={{ lineHeight: 1.8, color: 'var(--color-gray-700)', marginBottom: 24 }}>
                We present a modular pipeline for extracting structured data from restaurant menu images. 
                Our system combines optical character recognition (OCR) with multiple classification 
                approaches—rule-based heuristics, traditional machine learning models, and ensemble 
                methods—to produce schema-compliant JSON output without hallucinating structure not 
                present in the source image. We provide a comprehensive comparison of OCR backends 
                and evaluate six classification approaches on a hand-labeled test set.
              </p>
              
              <h2 style={{ fontSize: 20, marginBottom: 16 }}>Key Findings</h2>
              <ul style={{ lineHeight: 1.8, color: 'var(--color-gray-700)', marginBottom: 24, paddingLeft: 20 }}>
                <li><strong>Domain mismatch</strong> between receipt-trained models and menu images is the primary accuracy bottleneck</li>
                <li>ML models achieving <strong>91% accuracy on CORD-v2</strong> drop to <strong>&lt;30% F1 on menus</strong></li>
                <li>Domain-specific <strong>rule-based classification achieves 35.6% F1</strong> score and 38.9% price accuracy</li>
                <li><strong>GPU acceleration</strong> provides 3.5× speedup (333ms vs 1180ms per image)</li>
              </ul>
              
              <h2 style={{ fontSize: 20, marginBottom: 16 }}>Pipeline Architecture</h2>
              <div style={{ background: 'var(--color-gray-50)', padding: 16, borderRadius: 8, fontFamily: 'var(--font-mono)', fontSize: 13, marginBottom: 24 }}>
                Image → OCR → ColumnDetector → HybridClassifier → HierarchyFSM → PriceItemMatcher → JSON
              </div>
              
              <h2 style={{ fontSize: 20, marginBottom: 16 }}>Results Summary</h2>
              <div className="table-container">
                <table style={{ width: '100%', marginBottom: 24 }}>
                  <thead>
                    <tr>
                      <th>Approach</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>F1</th>
                      <th>Price Acc</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style={{ background: 'var(--color-gray-50)' }}>
                      <td><strong>Rule-Based</strong></td>
                      <td>41.3%</td>
                      <td>31.8%</td>
                      <td><strong>35.6%</strong></td>
                      <td>38.9%</td>
                    </tr>
                    <tr>
                      <td>Random Forest</td>
                      <td>34.6%</td>
                      <td>20.9%</td>
                      <td>25.6%</td>
                      <td>15.9%</td>
                    </tr>
                    <tr>
                      <td>XGBoost</td>
                      <td>37.8%</td>
                      <td>22.5%</td>
                      <td>27.7%</td>
                      <td>13.6%</td>
                    </tr>
                    <tr>
                      <td>MLP</td>
                      <td>43.9%</td>
                      <td>28.3%</td>
                      <td>34.0%</td>
                      <td>24.2%</td>
                    </tr>
                    <tr style={{ background: 'var(--color-gray-50)' }}>
                      <td><strong>Ensemble</strong></td>
                      <td>50.4%</td>
                      <td>27.5%</td>
                      <td>35.1%</td>
                      <td><strong>47.2%</strong></td>
                    </tr>
                  </tbody>
                </table>
              </div>
              
              <div style={{ textAlign: 'center', padding: 24, background: 'var(--color-gray-50)', borderRadius: 8 }}>
                <p style={{ marginBottom: 16, color: 'var(--color-gray-600)' }}>
                  For the complete paper with methodology details, equations, and analysis:
                </p>
                <a 
                  href="/paper/menu_ocr_paper.pdf" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="btn btn-primary"
                >
                  <FileText size={16} />
                  View Full PDF Paper
                </a>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Research Log */}
      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-header">
          <span className="card-title">Research Log</span>
        </div>
        <div className="card-body" style={{ maxHeight: 300, overflowY: 'auto' }}>
          <ResearchLogEntry 
            date="2024-12-15"
            title="v2.0 Pipeline Release"
            description="Implemented bipartite matching, hierarchy FSM, and column detection. F1 improved from 28% to 35.6%."
          />
          <ResearchLogEntry 
            date="2024-12-10"
            title="Domain Mismatch Analysis"
            description="Identified that CORD-trained models underperform on menus. ML models achieving 91% on receipts drop to <30% F1 on menus."
          />
          <ResearchLogEntry 
            date="2024-12-05"
            title="OCR Backend Comparison"
            description="Evaluated EasyOCR vs Tesseract. EasyOCR selected for GPU support (3.5× speedup) and API stability."
          />
          <ResearchLogEntry 
            date="2024-12-01"
            title="Initial Classifier Training"
            description="Trained Random Forest, XGBoost, MLP on CORD-v2 dataset. All achieved ~91% accuracy on validation set."
          />
        </div>
      </div>
    </div>
  );
}

function ResearchLogEntry({ date, title, description }) {
  return (
    <div style={{ 
      padding: '12px 0', 
      borderBottom: '1px solid var(--color-gray-100)',
      display: 'flex',
      gap: 16
    }}>
      <div style={{ 
        fontSize: 12, 
        color: 'var(--color-gray-400)', 
        whiteSpace: 'nowrap',
        paddingTop: 2
      }}>
        {date}
      </div>
      <div>
        <div style={{ fontWeight: 500, marginBottom: 4 }}>{title}</div>
        <div style={{ fontSize: 13, color: 'var(--color-gray-600)', lineHeight: 1.5 }}>
          {description}
        </div>
      </div>
    </div>
  );
}
