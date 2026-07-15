using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CyclonApp.Model
{
        public class ExportRevisionRow
    {
        public int Id { get; set; }
        public int RevisionNumber { get; set; }
        public string? RevisionNote { get; set; }
        public DateTime CreatedAt { get; set; }
        public bool HasResults { get; set; }
    }
}
