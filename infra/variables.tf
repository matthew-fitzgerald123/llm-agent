variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "llm-agent"
}

variable "db_instance_class" {
  type    = string
  default = "db.t3.micro"
}

variable "db_name" {
  type    = string
  default = "llm_agent"
}

variable "db_username" {
  type    = string
  default = "llm_agent"
}

variable "db_password" {
  type      = string
  sensitive = true
}

variable "p2_api_url" {
  type    = string
  default = "http://ml-platform-alb.example.com"
}

variable "p3_api_url" {
  type    = string
  default = "http://drift-monitor-alb.example.com"
}

variable "p4_api_url" {
  type    = string
  default = "http://rag-pipeline-alb.example.com"
}

variable "gen_model" {
  type    = string
  default = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
}
