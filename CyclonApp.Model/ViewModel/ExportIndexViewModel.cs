using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CyclonApp.Database;

namespace CyclonApp.Model.ViewModel
{
    public class ExportIndexViewModel
    {

        public CycloneDesign? Design { get; set; }

        public int? DesignId { get; set; }

        public List<ExportLog> ExportLogs { get; set; } = new();
    }
}
