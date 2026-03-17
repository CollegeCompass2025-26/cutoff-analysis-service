-- Database Schema for Cutoff Analysis Service

-- Core table for all historical cutoff data
CREATE TABLE raw_cutoffs (
    id SERIAL PRIMARY KEY,
    exam_name VARCHAR(50) NOT NULL, -- JoSAA, NEET, MHT-CET, etc.
    year INTEGER NOT NULL,
    round VARCHAR(20) NOT NULL,
    institute_type VARCHAR(50), -- IIT, NIT, Medical, etc.
    college_name TEXT NOT NULL,
    course_name TEXT NOT NULL,
    specialization VARCHAR(255),
    category VARCHAR(50) NOT NULL, -- GEN, OBC, SC, ST, etc.
    quota VARCHAR(50), -- AI, HS, OS, etc.
    gender VARCHAR(20),
    opening_rank INTEGER,
    closing_rank INTEGER,
    percentile FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- College Metadata
CREATE TABLE colleges_metadata (
    college_id SERIAL PRIMARY KEY,
    college_name TEXT UNIQUE NOT NULL,
    institute_type VARCHAR(50), -- IIT, NIT, Medical, etc.
    state VARCHAR(50),
    city VARCHAR(50),
    nirf_ranking INTEGER,
    avg_placement_package FLOAT,
    total_seats INTEGER,
    fees_per_year INTEGER
);

-- Indexing for fast retrieval
CREATE INDEX idx_cutoff_search ON raw_cutoffs (exam_name, year, college_name, course_name);
CREATE INDEX idx_cutoff_ranks ON raw_cutoffs (closing_rank);
