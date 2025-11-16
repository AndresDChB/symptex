from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Date, Float
from sqlalchemy.orm import relationship
from app.db.db import Base
import datetime

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, index=True)
    patient_file_id = Column(Integer, ForeignKey('patient_files.id'))
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    messages = relationship("ChatMessage", back_populates="session")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey('chat_sessions.id'))
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    session = relationship("ChatSession", back_populates="messages")

class PatientFile(Base):
    __tablename__ = "patient_files"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    birth_date = Column(Date)
    height = Column(Integer)
    weight = Column(Float)
    gender_identity = Column(String)
    gender_medical = Column(String)
    ethnic_origin = Column(String)
    anamneses = relationship("Anamnesis", back_populates="patient_file")
    anam_docs = relationship("AnamDoc", back_populates="patient_file")

class Anamnesis(Base):
    __tablename__ = "anamneses"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String)
    answer = Column(String)
    patient_file_id = Column(Integer, ForeignKey("patient_files.id"))
    patient_file = relationship("PatientFile", back_populates="anamneses")

class AnamDoc(Base):
    __tablename__ = "anam_docs"
    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String)
    type = Column(String)
    patient_file_id = Column(Integer, ForeignKey("patient_files.id"))
    patient_file = relationship("PatientFile", back_populates="anam_docs")
    description = Column(Text)
