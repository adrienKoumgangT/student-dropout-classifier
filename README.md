This project is the one made in the card of the evaluation of the Data Mining and Machine Learning course at the University of Pise.


# 1. Project Proposal

## 1.1 Domain and Core Idea

Student dropout is a complex, multifaceted problem that cannot be explained by academic performance alone.
This project aims to build an end-to-end machine learning pipeline to predict a student’s final academic status 
(Dropout, Enrolled, or Graduate) at the time of enrollment and after their first two semesters.
By leveraging a rich dataset that combines academic history, demographic background, socio-economic status,
and real-time macroeconomic indicators (GDP, Inflation), the core idea is to move beyond basic correlations
and use advanced classification algorithms and Explainable AI (XAI) to uncover the hidden, non-linear drivers of student attribution.

## 1.2 Dataset

- Name: Student's Dropout and Academic Success
- Source-Url: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- Description: This dataset contains 4.424 student records with 36 distinct features and one multi-class target variable.
It is designed to evaluate student outcomes holistically, moving beyond just academic grades to include external life factors.
Features Domains (36 Variables):
  - Demographics: Age, gender, nationality, and displacement status at enrollment.
  - Socio-Economic Background: Parent’s education and occupation, scholarship status, and financial health (e.g., debt and tuition payment status)
  - Macroeconomic Context: Regional unemployment, inflation, and GDP during the student’s academic tenure
  - Academic Performance: Admission metrics and detailed performance tracking across the first and second semesters

Target Variable (Target) is the final academic outcome of the student, divided into three categories: Graduate, Dropout, and Enrolled

## 1.3 ML Tasks

Classification.

## 1.4 Motivation

Every year, thousands of students leave college before finishing their degree.
But why? Is it because they cannot afford tuition? 
Because they have family or work responsibilities? 
Or because the economy rising prices, lack of jobs makes it too hard to continue? 
The truth is most students who drop out do not want to leave.
They are forced out by problems outside the classroom, and many regret it later.
This project aims to find out what really causes students to drop out and use that knowledge to build system 
that can predict right from the moment a student enrolls whether they are likely to drop out, graduate, or stay enrolled.
That way, schools can step in early and help before it is too late.



