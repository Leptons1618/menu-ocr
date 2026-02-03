import React from 'react';
import { Link, NavLink } from 'react-router-dom';
import { Upload, Image, GitCompare, FileText, BookOpen } from 'lucide-react';

export default function Header() {
  return (
    <header className="header">
      <div className="container header-content">
        <Link to="/" className="logo">
          <FileText size={24} />
          <h1>Menu OCR</h1>
        </Link>
        
        <nav className="nav">
          <NavLink 
            to="/" 
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          >
            <Upload size={16} />
            Upload
          </NavLink>
          <NavLink 
            to="/result" 
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          >
            <Image size={16} />
            Results
          </NavLink>
          <NavLink 
            to="/compare" 
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          >
            <GitCompare size={16} />
            Compare
          </NavLink>
          <NavLink 
            to="/research" 
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          >
            <BookOpen size={16} />
            Research
          </NavLink>
        </nav>
      </div>
    </header>
  );
}
