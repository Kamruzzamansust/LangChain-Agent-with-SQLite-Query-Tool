from langchain.tools import Tool
from langchain.pydantic_v1 import BaseModel
from langchain.tools import StructuredTool
from pydantic import BaseModel
from langchain.tools import Tool

def write_report(filename,html):
    with open(filename,'w') as f :
        f.write(html)

class  WriteReportArgsSchema(BaseModel):
    filename: str
    html: str

write_report_tool = StructuredTool.from_function(
    name = "Write_Report",
    description = " Write an HTML file to disk , use this to Use this tool when someone asks for a report ",
    func =  write_report,
    args_schema=WriteReportArgsSchema 
)